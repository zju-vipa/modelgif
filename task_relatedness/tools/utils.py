from functools import reduce
import saliency
from skimage import feature
from saliency.core import VisualizeImageGrayscale
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess
from captum.attr import visualization as viz, IntegratedGradients
import numpy as np
from captum.attr import InputXGradient
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def decode_segmap(image, nc=17):
    label_colors = np.array([
        (0, 0, 0),  # 0=background
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        (0, 128, 128),
        (128, 128, 128),
        (64, 0, 0),
        (192, 0, 0),
        (64, 128, 0),
        # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
        (192, 128, 0),
        (64, 0, 128),
        (192, 0, 128),
        (64, 128, 128),
        (192, 128, 128),
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        (0, 64, 0),
        (128, 64, 0)
    ])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def normalize_image(x):
    x = np.array(x).astype(np.float32)
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def abs_grayscale_norm(img):
    """Returns absolute value normalized image 2D."""
    assert isinstance(img, np.ndarray), "img should be a numpy array"
    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        img = np.absolute(img)
        img = img / float(img.max())
    else:
        img = VisualizeImageGrayscale(img)
    return img


def diverging_norm(img):  # [-1,1]
    """Returns image with positive and negative values."""
    assert isinstance(img, np.ndarray), "img should be a numpy array"
    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        imgmax = np.absolute(img).max()
        img = img / float(imgmax)
    else:
        img = saliency.VisualizeImageDiverging(img)
    return img


def get_field_cosdis(att1, att2, is_ig=False):
    if is_ig:
        att1, att2 = att1.flatten(1), att2.flatten(1)
    # cos dis
    cur_dis = 1 - torch.cosine_similarity(
        att1, att2, dim=2 if not is_ig else 1)
    result = (torch.mean(cur_dis)).numpy()
    return result


def scalestd(att, is_ig=False):
    att = att.cuda()
    if is_ig:
        att = (att - att.mean(0, keepdim=True)) / att.std(0, keepdim=True)
    else:
        att = (att - att.mean((0, 1), keepdim=True)) / att.std(
            (0, 1), keepdim=True)
    return att.cpu()


def get_field_cosdis_norm(att1, att2, is_ig=False):
    if is_ig:
        att1, att2 = att1.flatten(1), att2.flatten(1)
    att1, att2 = scalestd(att1), scalestd(att2)
    # cos dis
    cur_dis = 1 - torch.cosine_similarity(
        att1, att2, dim=2 if not is_ig else 1)
    result = (torch.mean(cur_dis)).numpy()
    return result


def get_field_eucldis(att1, att2, is_ig=False):
    if is_ig:
        att1, att2 = att1.flatten(1), att2.flatten(1)
    # dis mean
    distance = torch.sqrt(
        torch.sum(torch.square(att1 - att2), dim=2 if not is_ig else 1))
    result = torch.mean(distance).numpy()
    return result


def get_field_euclfrechet(att1, att2, is_ig=False):
    if is_ig:
        att1, att2 = att1.flatten(1), att2.flatten(1)
    distance = torch.sqrt(
        torch.sum(torch.square(att1 - att2), dim=2 if not is_ig else 1))
    max_result, _ = torch.max(distance, dim=1)
    result = torch.mean(max_result).numpy()
    return result


def get_field_cosfrechet(att1, att2, is_ig=False):
    if is_ig:
        att1, att2 = att1.flatten(1), att2.flatten(1)
    cos_dis = 1 - torch.cosine_similarity(
        att1, att2, dim=2 if not is_ig else 1)
    cur_dis, _ = torch.max(cos_dis, dim=1)
    result = (torch.mean(cur_dis)).numpy()
    return result


def get_field_absdis(att1, att2):
    # dis mean
    distance = torch.abs(att1 - att2)
    result = torch.mean(distance).numpy()
    return result


def get_distance_spr(att1, att2):
    # att1 = torch.nn.functional.relu(att1, inplace=True)
    # att2 = torch.nn.functional.relu(att2, inplace=True)
    cos_sim_sum = 0.0
    n = len(att1)
    sum = 0
    for i in range(n):
        # # ig: 0.699 0.737 | s: 0.51 0.567 | input 0.72 0.734
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # att1_i = abs_grayscale_norm(att1_i)
        # att2_i = abs_grayscale_norm(att2_i)
        # cos_sim, _ = spearmanr(att1_i.flatten(), att2_i.flatten())

        # # ig: 0.775 0.743 | s 0.51 0.567
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # att1_i = diverging_norm(att1_i)
        # att2_i = diverging_norm(att2_i)
        # cos_sim, _ = spearmanr(att1_i.flatten(), att2_i.flatten())

        # # ig:  0.793 0.829  | s 0.516 0.563  | inputXgradient 0.762 0.81
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # cos_sim, _ = spearmanr(att1_i.flatten(), att2_i.flatten())
        # cos_sim = cos_sim.__abs__()

        # ig:  0.793 0.829  | s 0.516 0.563  | inputXgradient 0.762 0.81
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # cos_sim, _ = spearmanr(att1_i.flatten(), att2_i.flatten())
        # # cos_sim = cos_sim.__abs__()

        # ig 0.698 0.733  |s 0.633 0.64
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # att1_i = abs_grayscale_norm(att1_i)
        # att2_i = abs_grayscale_norm(att2_i)
        # att1_i = torch.from_numpy(att1_i).flatten()
        # att2_i = torch.from_numpy(att2_i).flatten()
        # temp_cos_sim = torch.dot(att1_i, att2_i)  # tensor 1
        # cos_sim = (temp_cos_sim / (torch.norm(att1_i) * torch.norm(att2_i))).cpu().detach().numpy()

        # # ig 0.34 0.312 | s 0.624 0.605 | inputX 0.068 0.091
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # att1_i = diverging_norm(att1_i)
        # att2_i = diverging_norm(att2_i)
        # att1_i = torch.from_numpy(att1_i).flatten()
        # att2_i = torch.from_numpy(att2_i).flatten()
        # temp_cos_sim = torch.dot(att1_i, att2_i)  # tensor 1
        # cos_sim = (temp_cos_sim / (torch.norm(att1_i) * torch.norm(att2_i))).cpu().detach().numpy()

        # # ig 0.637 0.653
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # att1_i = np.abs(att1_i)
        # att2_i = np.abs(att2_i)
        # att1_i = torch.from_numpy(att1_i).flatten()
        # att2_i = torch.from_numpy(att2_i).flatten()
        # temp_cos_sim = torch.dot(att1_i, att2_i)  # tensor 1
        # cos_sim = (temp_cos_sim / (torch.norm(att1_i) * torch.norm(att2_i))).cpu().detach().numpy()

        # # ig  0.806 0.824     |  s:0.619  0.609   | inputX 0.764 0.806
        # att1_flat = att1[i].flatten()  # tensor 49152
        # att2_flat = att2[i].flatten()  # tensor 49152
        # temp_cos_sim = torch.dot(att1_flat, att2_flat)  # tensor 1
        # cos_sim = (temp_cos_sim / (torch.norm(att1_flat) * torch.norm(att2_flat))).cpu().detach().numpy()
        # cos_sim = 0.5+0.5*cos_sim

        # used in paper
        att1_flat = att1[i].flatten()  # tensor 49152
        att2_flat = att2[i].flatten()  # tensor 49152
        temp_cos_sim = torch.dot(att1_flat, att2_flat)  # tensor 1
        cos_sim = (temp_cos_sim / max(torch.norm(att1_flat) * torch.norm(att2_flat), 1e-8)).cpu().detach().numpy()

        # ig 0.813 0.825 |
        # att1_flat = att1[i].flatten()  # tensor 49152
        # att2_flat = att2[i].flatten()  # tensor 49152
        # temp_cos_sim = torch.dot(att1_flat, att2_flat)  # tensor 1
        # temp_cos_sim = torch.nn.functional.relu(temp_cos_sim, inplace=True)
        # cos_sim = (temp_cos_sim / (torch.norm(att1_flat) * torch.norm(att2_flat))).cpu().detach().numpy()

        # # ig  0.367 0.339     | s:0.624 0.616
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # att1_i = normalize_image(att1_i)
        # att2_i = normalize_image(att2_i)
        # att1_i = torch.from_numpy(att1_i).flatten()
        # att2_i = torch.from_numpy(att2_i).flatten()
        # temp = torch.dot(att1_i, att2_i)  # tensor 1
        # cos_sim = (temp / (torch.norm(att1_i) * torch.norm(att2_i))).cpu().detach().numpy()
        #

        # ig  0.397 0.419  | s 0.482  0.558
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # att1_i = normalize_image(att1_i)
        # att2_i = normalize_image(att2_i)
        # cos_sim = ssim(att1_i, att2_i,
        #            gaussian_weights=True, multichannel=True)

        # ig 0.097 0.095 |  如果把ig relu一下，就是0.628 0.625  |  s   0.673 0.701
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # cos_sim = ssim(att1_i, att2_i,
        #            gaussian_weights=True, multichannel=True)

        # # 0.671 0.734  |  s 0.493 0.542
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # att1_i = normalize_image(att1_i)
        # att2_i = normalize_image(att2_i)
        # normal_hog = feature.hog(att1_i,
        #                          pixels_per_cell=(16, 16))
        # rand_hog = feature.hog(att2_i,
        #                        pixels_per_cell=(16, 16))
        # cos_sim = spearmanr(normal_hog, rand_hog)[0]

        # # ig 0.739 0.752
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # att1_i = normalize_image(att1_i)
        # att2_i = normalize_image(att2_i)
        # normal_hog = feature.hog(att1_i,
        #                          pixels_per_cell=(8, 8))
        # rand_hog = feature.hog(att2_i,
        #                        pixels_per_cell=(8, 8))
        # cos_sim = spearmanr(normal_hog, rand_hog)[0]

        #  # | s  0.493 0.542
        # att1_i = att1[i].cpu().permute(1, 2, 0).detach().numpy()
        # att2_i = att2[i].cpu().permute(1, 2, 0).detach().numpy()
        # normal_hog = feature.hog(att1_i,
        #                          pixels_per_cell=(16, 16))
        # rand_hog = feature.hog(att2_i,
        #                        pixels_per_cell=(16, 16))
        # cos_sim = spearmanr(normal_hog, rand_hog)[0]

        # # ig 0.731  0.743 | s 0.518 0.56
        # att1_flat = att1[i].flatten()  # tensor 49152
        # att2_flat = att2[i].flatten()  # tensor 49152
        # cos_sim, _ = spearmanr(att1_flat, att2_flat)

        # temp = torch.dot(att1_flat, att2_flat)  # tensor 1
        #
        # if temp_cos_sim.item() < 0:
        #     sum = sum + 1
        # temp_cos_sim = torch.nn.functional.relu(temp_cos_sim, inplace=True)
        #
        # print(m.eq(temp_cos_sim))
        #
        # cos_sim = (temp_cos_sim / (torch.norm(att1_flat) * torch.norm(att2_flat))).cpu().detach().numpy()
        cos_sim_sum = cos_sim_sum + cos_sim
    # print(sum)
    cos_sim_sum = max(cos_sim_sum, 1e-8)
    return (n / cos_sim_sum)
    # return (n / cos_sim_sum).__abs__()
    # print((n / cos_sim_sum))
    # return (1- cos_sim_sum/n)


def get_distance(att1, att2):
    # 0.8
    cos_sim_sum = 0.0
    n = len(att1)
    sum = 0
    for i in range(n):
        att1_flat = att1[i].flatten()  # tensor 49152
        att2_flat = att2[i].flatten()  # tensor 49152
        temp_cos_sim = torch.dot(att1_flat, att2_flat)  # tensor 1
        if temp_cos_sim.item() < 0:
            sum = sum + 1
        temp_cos_sim = torch.nn.functional.relu(temp_cos_sim, inplace=True)
        cos_sim = (temp_cos_sim / (torch.norm(att1_flat) * torch.norm(att2_flat))).cpu().detach().numpy()
        cos_sim_sum = cos_sim_sum + cos_sim
    # print(sum)
    return n / cos_sim_sum
    # cos_sim_sum = 0.0
    # n = len(att1)
    # for i in range(n):
    #
    #     att1_flat = att1[i].flatten()
    #     att2_flat = att2[i].flatten()
    #     temp_cos_sim = torch.dot(att1_flat, att2_flat)
    #     temp_cos_sim = 0.5 + 0.5 * temp_cos_sim
    #     att1_norm = 0.5 + 0.5 * torch.norm(att1_flat)
    #     att2_norm = 0.5 + 0.5 * torch.norm(att2_flat)
    #     cos_sim = (temp_cos_sim / (att1_norm * att2_norm)).cpu().detach().numpy()
    #     cos_sim_sum = cos_sim_sum + cos_sim
    #
    # return n/cos_sim_sum


def get_distance_new(att1, att2):
    # att1 =att1[data < 0] = 0
    #
    # att1 = torch.nn.functional.relu(att1, inplace=True)
    # att2 = torch.nn.functional.relu(att2, inplace=True)

    cos_sim_sum = 0.0
    n = len(att1)
    sum = 0
    for i in range(n):
        att1_flat = att1[i].flatten()  # tensor 49152
        att2_flat = att2[i].flatten()  # tensor 49152
        tempa = torch.norm(att1_flat)
        tempb = torch.norm(att2_flat)
        mina = torch.min(att1_flat)
        minb = torch.min(att2_flat)
        maxa = torch.max(att1_flat)
        maxb = torch.max(att2_flat)

        temp_cos_sim = torch.dot(att1_flat, att2_flat)  # tensor 1

        if temp_cos_sim.item() < 0:
            sum = sum + 1
        # temp_cos_sim = torch.nn.functional.relu(temp_cos_sim, inplace=True)

        # print(m.eq(temp_cos_sim))

        cos_sim = (temp_cos_sim /
                   (torch.norm(att1_flat) *
                    torch.norm(att2_flat))).cpu().detach().numpy()
        cos_sim_sum = cos_sim_sum + cos_sim
    print(sum)
    return n / cos_sim_sum


# def get_distance_new(att1, att2):  # 128,3,128,128
#     cos_sim_sum = 0.0
#     n = len(att1)
#     sum = 0
#     w = att1.shape[2]
#     h = att1.shape[3]
#
#
#     # for i in range(3*w*h):
#     #     for img_i in range(n):
#     #         img_flat = att1[img_i].flatten()
#     #         a = img_flat[i].unsqueeze_(0)
#     #
#     #         if img_i == 0:
#     #             aa = a
#     #         else:
#     #             aa = torch.cat((aa, a), dim=0)
#     #     aa = aa.unsqueeze_(0)
#     #     if i ==0:
#     #         bb =aa
#     #         # bb = bb
#     #     else:
#     #         bb = torch.cat((bb, aa), 0)
#     # print("ok")
#
#
#     att1_new = att1.view(n,-1)  # 6,49152
#     att2_new = att2.view(n,-1)  # 6,49152
#     att1_new = att1_new.cpu().permute(1,0)
#     att2_new = att2_new.cpu().permute(1,0)
#
#     nn = len(att1_new)
#     for i in range(nn):
#         att1_flat = att1_new[i].flatten()  # tensor 49152
#         att2_flat = att2_new[i].flatten()  # tensor 49152
#         temp_cos_sim = torch.dot(att1_flat, att2_flat)  # tensor 1
#         temp_cos_sim = torch.nn.functional.relu(temp_cos_sim, inplace=True)
#         weigt = temp_cos_sim/ (torch.norm(att1_flat) * torch.norm(att2_flat)).unsqueeze_(0)
#         if i == 0:
#             weights = weigt
#         else:
#             weights = torch.cat((weights, weigt), dim=0)
#         cos_sim = (temp_cos_sim/ (torch.norm(att1_flat) * torch.norm(att2_flat))).cpu().detach().numpy()
#         cos_sim_sum = cos_sim_sum + cos_sim
#
#
#     print(sum)
#     print(nn/cos_sim_sum)
#     return nn/cos_sim_sum


def get_distance_new_weight(att1, att2):  # 128,3,128,128
    n = len(att1)
    att1_new = att1.view(n, -1)  # 6,49152
    att2_new = att2.view(n, -1)  # 6,49152
    att1_new = att1_new.cpu().permute(1, 0)
    att2_new = att2_new.cpu().permute(1, 0)
    nn = len(att1_new)
    for i in range(nn):
        att1_flat = att1_new[i].flatten()  # tensor 49152
        att2_flat = att2_new[i].flatten()  # tensor 49152
        temp_cos_sim = torch.dot(att1_flat, att2_flat)  # tensor 1
        temp_cos_sim = torch.nn.functional.relu(temp_cos_sim, inplace=True)
        weigt = temp_cos_sim / (torch.norm(att1_flat) * torch.norm(att2_flat)).unsqueeze_(0)  # 越小越相似
        if i == 0:
            weights = weigt
        else:
            weights = torch.cat((weights, weigt), dim=0)
    weights = torch.tensor(weights, dtype=torch.float32)
    # print("weights")
    # # 越相似，权重越大
    # weights_min = weights.min()
    # weights_max = weights.max()
    # if weights_max - weights_min ==0:
    #     weights = torch.ones_like(weights)
    # else:
    #     weights = (weights_max - weights) / (weights_max - weights_min)
    # # weights[-1,1] 越大越相似，越相似，权重越高
    # weights_min = weights.min()
    # weights_max = weights.max()
    # # weights = 1-weights  # 范围是[0,2]
    # if weights_max - weights_min == 0 :
    #     weights = torch.ones_like(weights)
    # else:
    #     weights = (weights-weights_min) / (weights_max - weights_min)
    # weights[-1,1] 越大越相似，越相似，权重越小
    weights_min = weights.min()
    weights_max = weights.max()
    # weights = 1-weights  # 范围是[0,2]
    if weights_max - weights_min == 0:
        weights = torch.zeros_like(weights)
    else:
        weights = (weights_max - weights) / (weights_max - weights_min)
    cos_sim_sum = 0.0
    sum = 0
    for i in range(n):
        att1_flat = att1[i].flatten()  # tensor 49152
        att2_flat = att2[i].flatten()  # tensor 49152
        att1_flat = torch.tensor(att1_flat, dtype=torch.float32)
        att2_flat = torch.tensor(att2_flat, dtype=torch.float32)
        # weights = weights.unsqueeze_(0)
        # att1_flat = att1_flat.unsqueeze_(0)
        # att2_flat = att2_flat.unsqueeze_(0)
        # temp_cos_sim = torch.matmul(att1_flat, att2_flat,weights )  # tensor 1
        # temp_cos_sim = att1_flat@att2_flat@weights
        # temp_cos_sim = att1_flat.dot(weights)
        # temp_cos_sim = np.linalg.multi_dot([att1_flat, att2_flat, weights])
        # matrixs = [att1_flat, att2_flat,weights]
        #
        # temp_cos_sim = reduce(np.dot, matrixs)
        temp_cos_sim = (att1_flat * att2_flat * weights).sum()
        if temp_cos_sim.item() < 0:
            sum = sum + 1
        temp_cos_sim = torch.nn.functional.relu(temp_cos_sim, inplace=True)
        # print(m.eq(temp_cos_sim))
        cos_sim = (temp_cos_sim /
                   (torch.norm(att1_flat) *
                    torch.norm(att2_flat))).cpu().detach().numpy()
        cos_sim_sum = cos_sim_sum + cos_sim
    # print(sum)
    # print(n / cos_sim_sum)
    # print(n / cos_sim_sum)
    if cos_sim_sum == 0:
        return 1
    else:
        return n / cos_sim_sum
