import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch Image Training')
parser.add_argument('-r', '--root', default='./run', help='root dir')
parser.add_argument('-n',
                    '--name',
                    default='field_cosdis',
                    help='task name')
parser.add_argument('-p', '--path', default='temp', help='config name')
parser.add_argument('-g',
                    '--gpu',
                    default=None,
                    type=str,
                    help='GPU id to use.')
parser.add_argument('--sk', default=-1, type=int, help='skip mode')
parser.add_argument('-s', '--seed', default=0, type=int, help='seed')

args = parser.parse_args()
if args.gpu != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from dataset import dataset1, dataset4, dataset_field_sample
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from model_load import load_model
from sklearn.metrics import roc_curve, auc
from captum.attr import IntegratedGradients, GuidedBackprop
from field_generator import FieldGenerator
from tqdm import tqdm
import numpy as np
from torch import nn
from random import Random

BATCH_SIZE = 2

from log import get_logger

rootdir = args.root
os.makedirs(rootdir, exist_ok=True)
logger = get_logger(
    rootdir,
    f'{args.name}.log',
)
random_baseline = False

attribute_only_prediction = True
class_num = 10
skip_mode = args.sk  # 0:last half, 1 front half, 2: skip odd
seed = args.seed
n_steps = 50
dataset_name = [
    ('field_sample', 512, ["sort_by_distance_irr"]), # sort distance between irrelevant models and attacker models
    ('field_sample', 256, ["sort_by_distance_irr"]),
    ('field_sample', 128, ["sort_by_distance_irr"]),
    ('field_sample', 64, ["sort_by_distance_irr"]),
    ('field_sample', 32, ["sort_by_distance_irr"]),
    ('field_sample', 16, ["sort_by_distance_irr"]),
    ('field_sample', 8, ["sort_by_distance_irr"]),
]


def calculate_auc(list_a, list_b):
    l1, l2 = len(list_a), len(list_b)
    y_true, y_score = [], []
    for i in range(l1):
        y_true.append(0)
    for i in range(l2):
        y_true.append(1)
    y_score.extend(list_a)
    y_score.extend(list_b)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


class FeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()


def shuffle_and_check_difference(li: list):
    raw_li = li
    li = li.copy()
    rd = Random(seed)
    while any(filter(lambda x: x[0] == x[1], zip(raw_li, li))) != 0:
        rd.shuffle(li)
    return li


def field_distance(att1, att2):
    # trans attr dis improved
    cur_dis = 1 - torch.cosine_similarity(att1.cuda(), att2.cuda(), dim=2)
    result = torch.mean(cur_dis).cpu()
    return result


def pairwise_euclid_distance(A):
    sqr_norm_A = torch.unsqueeze(torch.sum(torch.pow(A, 2), dim=1), dim=0)
    sqr_norm_B = torch.unsqueeze(torch.sum(torch.pow(A, 2), dim=1), dim=1)
    inner_prod = torch.matmul(A, A.transpose(0, 1))
    tile1 = torch.reshape(sqr_norm_A, [A.shape[0], 1])
    tile2 = torch.reshape(sqr_norm_B, [1, A.shape[0]])
    return tile1 + tile2 - 2 * inner_prod


def correlation_dist(A):
    A = F.normalize(A, dim=-1)
    cor = pairwise_euclid_distance(A)
    cor = torch.exp(-cor)

    return cor


def cal_model_gif(model, dataloader, early_stop):
    model.eval()
    model = model.cuda()
    outputs = []
    ig = FieldGenerator(model)
    pbar = tqdm(enumerate(dataloader), total=min(len(dataloader), early_stop))
    for i, (x, y) in pbar:
        if i >= early_stop:
            break
        x = x.cuda()
        if attribute_only_prediction:
            y = y.cuda()  # normal
            y = torch.argmax(model(x), dim=1)  # enf
            target_list = [y]
        else:
            target_list = [j for j in range(class_num)]

        if random_baseline:
            shuffle_idx = shuffle_and_check_difference(
                [j for j in range(len(y))])
            baselines = x[shuffle_idx].clone()

        for j, y in enumerate(target_list):
            if random_baseline:
                output_ = ig.attribute(x,
                                       target=y,
                                       baselines=baselines,
                                       n_steps=n_steps).flatten(2)
            else:
                output_ = ig.attribute(x, target=y,
                                       n_steps=n_steps).flatten(2)

            if skip_mode == 0:
                output_ = output_[:, n_steps // 2:]
            elif skip_mode == 1:
                output_ = output_[:, :n_steps // 2]
            elif skip_mode == 2:
                output_ = output_[:, 1::2]

            if j == 0:
                output = output_
            else:
                output = output + output_

        outputs.append(output.cpu().detach())

    outputs = torch.cat(outputs, dim=0)

    model = model.cpu()
    return outputs


def output_to_label(output):
    shape = output.shape
    pred = torch.argmax(output, dim=1)
    preds = 0.01 * torch.ones(shape)

    for i in range(shape[0]):
        preds[i, pred[i]] = 1

    preds = torch.softmax(preds, dim=-1)

    # print(preds[0,:])
    return preds


def cal_correlation(models, dataset_name):

    # SAC-normal
    early_Stop = 9999999
    if isinstance(dataset_name, tuple):
        if dataset_name[0] == 'normal':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
            testset = torchvision.datasets.CIFAR10(root='./data',
                                                   train=False,
                                                   download=True,
                                                   transform=transform_test)
            train_loader = torch.utils.data.DataLoader(testset,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=False)
        elif dataset_name[0] == 'field_sample':
            testset = dataset_field_sample("reference_sample.h5",
                                           dataset_name[2])
            train_loader = DataLoader(testset,
                                      shuffle=False,
                                      batch_size=BATCH_SIZE)
        else:
            testset = dataset1(dataset_name[0], train=False)
            train_loader = DataLoader(testset,
                                      shuffle=False,
                                      batch_size=BATCH_SIZE)
        if dataset_name[1] > 0:
            early_Stop = dataset_name[1] // BATCH_SIZE
            dataset_name = f'{dataset_name[0]} {early_Stop*BATCH_SIZE}'
        else:
            dataset_name = f'{dataset_name[0]} {len(testset)}'
    else:
        testset = dataset1(dataset_name, train=False)
        train_loader = DataLoader(testset,
                                  shuffle=False,
                                  batch_size=BATCH_SIZE)
    logger.info(f"dataset name: {dataset_name}")

    diff = torch.zeros(len(models))
    attr0 = cal_model_gif(models[0], train_loader, early_Stop)
    for i in range(len(models) - 1):
        iter = i + 1
        print("Iter:", iter, "/", len(models))
        model = models[iter]
        attriter = cal_model_gif(model, train_loader, early_Stop)
        diff[i] = field_distance(attriter, attr0)

    logger.info(f"Correlation difference is: {diff[:20]}")
    logger.info(f"Correlation difference is: {diff[20:40]}")
    logger.info(f"Correlation difference is: {diff[40:60]}")
    logger.info(f"Correlation difference is: {diff[60:70]}")
    logger.info(f"Correlation difference is: {diff[70:80]}")
    logger.info(f"Correlation difference is: {diff[80:90]}")
    logger.info(f"Correlation difference is: {diff[90:100]}")
    logger.info(f"Correlation difference is: {diff[100:120]}")
    logger.info(f"Correlation difference is: {diff[120:135]}")
    logger.info(f"Correlation difference is: {diff[135:155]}")

    list1 = diff[:20]
    list2 = diff[20:40]
    list3 = diff[40:60]
    list4 = diff[60:70]
    list5 = diff[70:80]
    list6 = diff[80:90]
    list7 = diff[90:100]
    list8 = diff[100:120]
    list9 = diff[120:135]
    list10 = diff[135:155]

    auc_p = calculate_auc(list1, list3)
    auc_l = calculate_auc(list2, list3)
    auc_finetune = calculate_auc(list10, list3)
    auc_adv = calculate_auc(list8, list3)
    auc_prune = calculate_auc(list9[:10], list3)
    auc_100 = calculate_auc(list4, list5)
    auc_10C = calculate_auc(list6, list7)

    logger.info("Calculating AUC:")

    logger.info(
        f"AUC_P: {auc_p} AUC_L: {auc_l} AUC_Finetune: {auc_finetune} AUC_Prune: {auc_prune} AUC_Adv: {auc_adv} AUC_100: {auc_100} AUC_10C: {auc_10C}"
    )


if __name__ == '__main__':

    models = []

    for i in [0]:
        globals()['teacher' + str(i)] = load_model(i, "teacher")
        models.append(globals()['teacher' + str(i)])

    for i in range(20):
        globals()['student_kd' + str(i)] = load_model(i, "student_kd")
        models.append(globals()['student_kd' + str(i)])

    for i in range(20):
        globals()['student' + str(i)] = load_model(i, "student")
        models.append(globals()['student' + str(i)])

    for i in range(20):
        globals()['clean' + str(i)] = load_model(i, "irrelevant")
        models.append(globals()['clean' + str(i)])
    #

    for i in range(10):
        globals()['finetune' + str(i)] = load_model(i, "finetune-100")
        models.append(globals()['finetune' + str(i)])

    for i in range(10):
        globals()['CIFAR10C' + str(i)] = load_model(i, "CIFAR100")
        models.append(globals()['CIFAR10C' + str(i)])

    for i in range(10):
        globals()['finetune' + str(i)] = load_model(i, "finetune-10C")
        models.append(globals()['finetune' + str(i)])
    #
    for i in range(10):
        globals()['CIFAR10C' + str(i)] = load_model(i, "CIFAR10C")
        models.append(globals()['CIFAR10C' + str(i)])

    for i in range(20):
        globals()['adv' + str(i)] = load_model(i, "adv_train")
        models.append(globals()['adv' + str(i)])

    for i in range(15):
        globals()['fp' + str(i)] = load_model(i, "fine-pruning")
        models.append(globals()['fp' + str(i)])

    for i in range(20):
        globals()['finetune_normal' + str(i)] = load_model(
            i, 'finetune_normal')
        models.append(globals()['finetune_normal' + str(i)])
    model: nn.Module
    for model in models:
        for m in model.modules():
            if isinstance(m, (nn.ReLU, nn.LeakyReLU)):
                m.inplace = False
    for na in dataset_name:
        logger.info(na)
        cal_correlation(models, na)