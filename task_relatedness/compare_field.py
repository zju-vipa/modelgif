import math
import os

import argparse

parser = argparse.ArgumentParser(description='Attribution Map Comparison')
parser.add_argument('-r', '--root', default='./result_field_featuremap_tk', help='save root')
parser.add_argument('-p', '--path', default='result_1000', help='config name')
parser.add_argument('-g',
                    '--gpu',
                    default=None,
                    type=str,
                    help='Index of GPU to use.')
parser.add_argument('-d',
                    '--distance',
                    default=0,
                    type=int,
                    help='methods for distance comparison')
parser.add_argument('--ig',
                    action="store_true",
                    help='use integrated gradient')
parser.add_argument('--lh', action="store_true", help='last half')
parser.add_argument('--lg', default="default", type=str, help='log group')

args = parser.parse_args()
if args.gpu != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import pickle
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess
from torchvision import transforms

from captum.attr import visualization as viz, IntegratedGradients, LayerActivation, Saliency, \
    DeepLiftShap, DeepLift, NoiseTunnel
import numpy as np
from captum.attr import InputXGradient
import matplotlib.pyplot as plt
from tools.utils import get_field_cosdis, get_field_eucldis, get_field_euclfrechet, get_field_cosfrechet, get_field_cosdis_norm
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from log import get_logger

root_dir = args.root
path_name = os.path.join(root_dir, args.path)
# path_name = "result_new/result_smooth_new"
os.makedirs(os.path.join(root_dir, args.lg), exist_ok=True)
logger = get_logger(os.path.join(root_dir, args.lg), f'{args.path}.log')
out = 0
out_max = 0
target_num = 0
max_class = 0
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

device = "cuda"

# list of 17 task

task_name_map = {
    "autoencoding": "autoencoder",
    "curvature": "curvature",
    "denoising": "denoise",
    "edge_texture": "edge2d",
    "edge_occlusion": "edge3d",
    "keypoints2d": "keypoint2d",
    "keypoints3d": "keypoint3d",
    "reshading": "reshade",
    "depth_zbuffer": "rgb2depth",
    "depth_euclidean": "rgb2mist",
    "normal": "rgb2sfnorm",
    "room_layout": "room_layout",
    "segment_unsup25d": "segment25d",
    "segment_unsup2d": "segment2d",
    "vanishing_point": "vanishing_point_well_defined",
    "segment_semantic": "segmentsemantic_rb",
    "class_object": "class_1000"
}

#14task
# task_list_name = 'autoencoding curvature denoising edge_texture edge_occlusion \
# keypoints2d keypoints3d \
# reshading depth_zbuffer depth_euclidean normal \
# segment_unsup25d segment_unsup2d segment_semantic'

# 17 task
task_list_name = 'autoencoding curvature denoising edge_texture edge_occlusion \
keypoints2d keypoints3d \
reshading depth_zbuffer depth_euclidean normal \
room_layout segment_unsup25d segment_unsup2d vanishing_point \
segment_semantic class_object'

# 没实现的 room_layout vanishing_point class_object
# 3通道的 autoencoding denoising normal
distance_list = [
    get_field_cosdis, get_field_eucldis, get_field_cosfrechet,
    get_field_euclfrechet, get_field_cosdis_norm
]
task_list = task_list_name.split()
task_list_origin = [task_name_map[task_name] for task_name in task_list]
list_zero = [0 for i in range(17)]
task_dict = dict(zip(task_list, list_zero))
logger.info(path_name)
for key in task_dict:
    logger.info(key)

    if os.path.exists(f"{path_name}/{key}") and os.path.exists(
            f"{path_name}/{key}/{key}_att.npy"):
        temp = np.load(f"{path_name}/{key}/{key}_att.npy")
        if args.lh:
            logger.info("last half")
            temp = np.ascontiguousarray(temp[:, temp.shape[1] // 2:])
        elif temp.shape[0] > 1500 and not args.ig:
            temp = np.ascontiguousarray(temp[:, 1::2])
        task_dict[key] = torch.from_numpy(temp)
        logger.info("加载了")

        continue
    else:
        raise NotImplementedError
logger.info(task_dict["autoencoding"].shape)
affinity_matrix = np.zeros((len(task_dict), len(task_dict)), float)
index1 = -1
index2 = -1
for task1, task_att1 in task_dict.items():
    index1 = index1 + 1
    index2 = -1
    for task2, task_att2 in task_dict.items():
        index2 = index2 + 1
        # if index2 > 13:
        #     break
        if index1 > index2:
            continue
        # if index2 == 3:
        #     kkk = 5

        dis = distance_list[args.distance](task_att1, task_att2, is_ig=args.ig)
        logger.info(dis)
        affinity_matrix[index1, index2] = dis
        affinity_matrix[index2, index1] = dis
logger.info("Affinity Matrics")
logger.info(f'{affinity_matrix}')
np.save(os.path.join(root_dir, args.lg, f'{args.path}.npy'), affinity_matrix)

# Load affinity matrix
with open(r'./data/affinities/all_affinities.pkl', 'rb') as f:
    data = pickle.load(f)
t_affinity_matrix = np.zeros((len(task_list_origin), len(task_list_origin)))
for i, task1 in enumerate(task_list_origin):
    for j, task2 in enumerate(task_list_origin):
        t_affinity_matrix[i, j] = data[task1 + "__" + task2]

# corr_value_matrix0 = np.zeros(len(task_list) - 1)
corr_value_matrix0 = np.zeros(len(task_list))
for j, task in enumerate(task_list):
    # if j >= 13:
    #     break
    temp, _ = spearmanr(t_affinity_matrix[:, j], affinity_matrix[:, j])
    corr_value_matrix0[j] = temp
logger.info(corr_value_matrix0)
max_corr = 0
mean_corr0 = np.mean(corr_value_matrix0)
logger.info(path_name)
logger.info(f"Spearman correlation: {mean_corr0}")