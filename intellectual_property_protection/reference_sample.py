import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from model_load import load_model
import torchvision
import torch
from torchvision import transforms
from field_generator import FieldGenerator
from tqdm import tqdm
import h5py
import numpy as np

BATCH_SIZE = 32


def field_distance(att1, att2):
    att1, att2 = att1.cuda(), att2.cuda()
    # trans attr dis improved
    cur_dis = 1 - torch.cosine_similarity(att1, att2, dim=2)
    result = torch.mean(cur_dis, dim=1).cpu()
    return result


n_steps = 50

if __name__ == '__main__':

    models = []
    teacher = load_model(0, "teacher")
    for i in tqdm(range(6), total=6):
        globals()['student_model_tk_' + str(i)] = load_model(i, "teacher_kd")
        models.append(globals()['student_model_tk_' + str(i)])
    for i in tqdm(range(20), total=20):
        globals()['clean' + str(i)] = load_model(i, "irrelevant")
        models.append(globals()['clean' + str(i)])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform_test)
    train_loader = torch.utils.data.DataLoader(testset,
                                               num_workers=8,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False)

    imgs = []
    labels = []
    attr_scores = []
    preds = []
    student_preds = []
    teacher.eval()
    teacher.cuda()
    teacher_ig = FieldGenerator(teacher)
    for i, (img,
            target) in enumerate(tqdm(train_loader, total=len(train_loader))):
        # if i > 3:
        #     break
        x = normalization(img).cuda()
        y = torch.argmax(teacher(x), dim=1)
        attr_teacher = teacher_ig.attribute(x, target=y).flatten(2)
        preds.append(y.cpu().detach())
        student_pred = []
        dis = []
        for model in models:
            model.eval()
            model.cuda()
            ig = FieldGenerator(model)
            y = torch.argmax(model(x), dim=1)
            student_pred.append(y.cpu().detach())
            attr = ig.attribute(x, target=y).flatten(2)
            dis_ = field_distance(attr_teacher, attr)
            dis.append(dis_.cpu().detach())
            model.cpu()
        attr_scores.append(torch.stack(dis, dim=1))
        imgs.append(img)
        labels.append(target)
        student_preds.append(torch.stack(student_pred, dim=1))

    attr_scores = torch.cat(attr_scores)
    imgs = torch.cat(imgs)
    labels = torch.cat(labels)
    preds = torch.cat(preds)
    student_preds = torch.cat(student_preds)

    # _, idx = torch.sort(attr_scores, descending=True)
    # for s in [attr_scores, imgs, labels, preds, student_preds]:
    #     s[:] = s[idx]

    # print(attr_scores)

    file1 = h5py.File('data/reference_sample.h5', 'w')
    # print(attr_scores.shape)

    file1.create_dataset("/data", data=np.array(imgs))
    file1.create_dataset("/label", data=np.array(labels))
    file1.create_dataset("/distance", data=np.array(attr_scores))
    file1.create_dataset("/preds", data=np.array(preds))
    file1.create_dataset("/student_preds", data=np.array(student_preds))
