import argparse
import os
from field_generator import FieldGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--rs', type=str, default="default", help="result dir")
parser.add_argument('--model1', type=str, default="resnet20")
parser.add_argument('--path1', type=str, default="ckpt_CIFAR10_1")
parser.add_argument('--model2', type=str, default="resnet20")
parser.add_argument('--path2', type=str, default="ckpt_CIFAR10_2")
parser.add_argument('--fix-epoch', type=int, default=-1)
parser.add_argument('--nc', type=int, default=10, help="num classes")
parser.add_argument('-g',
                    '--gpu',
                    default=None,
                    type=str,
                    help='GPU id to use.')
arg = parser.parse_args()
if arg.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu

import numpy as np
import utils
import train
import model
import torch
import torchvision
from torch.utils.data import DataLoader
from log import get_logger
from tqdm import tqdm

arg.path1 = os.path.join("models", arg.path1)
arg.path2 = os.path.join("models", arg.path2)
arg.rs = os.path.join("results", arg.rs)
os.makedirs(arg.rs, exist_ok=True)
logger = get_logger(
    arg.rs,
    f'distance.log',
)


def compute_attr(net, path, dataloader):
    states = torch.load(path, map_location="cpu")
    net.load_state_dict(states['net'])
    net.eval()
    net.cuda()
    attrs = []

    # def w(img):
    #     output = net(img)
    #     return output.sum(1)

    ig = FieldGenerator(net)
    for x, y in tqdm(dataloader, total=len(dataloader)):
        x = x.cuda()
        y = y.cuda()
        # y = net(x)
        # y = torch.argmax(y, dim=1).detach()

        # attr = ig.attribute(x, target=y).flatten(2)
        attr = ig.attribute(x, target=y).flatten(2)
        # attr = (attr - attr.mean(2, keepdim=True)) / attr.std(2, keepdim=True)
        attrs.append(attr.cpu().detach())
    net.cpu()
    attrs = torch.cat(attrs)
    return attrs


def calc_distance(attr1, attr2):
    # trans attr dis improved
    cur_dis = 1 - torch.cosine_similarity(attr1, attr2, dim=2)
    result = torch.mean(cur_dis).cpu().item()
    return result


def compare(path1, path2, dataloader):
    try:
        architecture_1 = eval(f"model.{arg.model1}")
    except:
        architecture_1 = eval(f"torchvision.models.{arg.model1}")
    try:
        architecture_2 = eval(f"model.{arg.model2}")
    except:
        architecture_2 = eval(f"torchvision.models.{arg.model2}")
    net1 = architecture_1()
    net2 = architecture_2()
    if arg.nc != 10:
        net1.linear = torch.nn.Linear(64, arg.nc)
        net2.linear = torch.nn.Linear(64, arg.nc)
    attr1 = compute_attr(net1, path1, dataloader)
    attr2 = compute_attr(net2, path2, dataloader)

    return calc_distance(attr1, attr2)


def main():
    probe_data = utils.load_dataset(arg.dataset,
                                    True,
                                    download=True,
                                    unlearning_drop=True,
                                    num_classes=arg.nc,
                                    is_probe=True)

    dataloader = DataLoader(probe_data,
                            batch_size=4,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)
    results = []
    for e in range(2, 201, 2):
        path1 = os.path.join(
            arg.path1,
            f"model_epoch_{e if arg.fix_epoch < 0 else arg.fix_epoch}")
        path2 = os.path.join(arg.path2, f"model_epoch_{e}")
        logger.info(f"comparing {path1} and {path2}")
        distance = compare(path1, path2, dataloader)
        logger.info(f"epoch {e} distance: {distance}")
        results.append(distance)
    results = np.array(results)
    logger.info(results)
    np.save(os.path.join(arg.rs, "distance.npy"), results)


if __name__ == "__main__":
    main()
