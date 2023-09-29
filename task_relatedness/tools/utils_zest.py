import copy
import os
import random

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from captum.attr import visualization as viz


# from sklearn.metrics.pairwise import cosine_similarity, paired_distances
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def get_parameters(net, numpy=False):
    # get weights from a torch model as a list of numpy arrays
    parameter = torch.cat([i.data.reshape([-1]) for i in list(net.parameters())])
    if numpy:
        return parameter.cpu().numpy()
    else:
        return parameter


def get_model(model, architecture, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    state = torch.load(model)
    net = architecture()
    net.load_state_dict(state['net'])
    net.to(device)
    return net


def get_distance(att1, att2):
    cos_sim_sum = 0.0
    n = len(att1)
    for i in range(n):
        att1_flat = att1[i].flatten()  # tensor 3072
        att2_flat = att2[i].flatten()  # tensor 3072
        cos_sim = (torch.dot(att1_flat, att2_flat) / (
                    torch.norm(att1_flat) * torch.norm(att2_flat))).cpu().detach().numpy()
        cos_sim_sum = cos_sim_sum + cos_sim
    return n / cos_sim_sum


def get_distance_input_x_gradient(att1, att2):
    cos_sim_sum = 0.0
    n = len(att1)
    for i in range(n):
        att1_flat = att1[i].flatten()  # tensor 3072
        att2_flat = att2[i].flatten()  # tensor 3072
        cos_sim = (torch.dot(att1_flat, att2_flat) / (
                    torch.norm(att1_flat) * torch.norm(att2_flat))).cpu().detach().numpy()
        cos_sim_sum = cos_sim_sum + cos_sim
    return n / cos_sim_sum


def parameter_distance(model1, model2, order=2, architecture=None, half=False, linear=False, lime=False):
    # compute the difference between 2 checkpoints

    weights1 = consistent_type(model1, architecture, half=half, linear=linear, lime=lime)  # 2046*10
    weights2 = consistent_type(model2, architecture, half=half, linear=linear, lime=lime)  # 2046*10
    if not isinstance(order, list):
        orders = [order]
    else:
        orders = order
    res_list = []
    if lime:
        temp_w1 = copy.copy(weights1)
        temp_w2 = copy.copy(weights2)
    for o in orders:
        if lime:
            weights1, weights2 = lime_align(temp_w1, temp_w2, o)
        res = compute_distance(weights1, weights2, o)
        if isinstance(res, np.ndarray):
            res = float(res)
        res_list.append(res)
    return res_list


def mean_std_to_array(mean, std, rgb_last=True):
    mean = np.array(mean)
    std = np.array(std)
    if rgb_last:
        mean = mean.reshape([1, 1, 1, -1])
        std = std.reshape([1, 1, 1, -1])
    else:
        mean = mean.reshape([1, -1, 1, 1])
        std = std.reshape([1, -1, 1, 1])
    return mean, std


def show_images(attribution, images):
    # 可视化
    attr_ig0 = np.transpose(attribution[0].squeeze().cpu().detach().numpy(), (1, 2, 0))  # 32,32,3
    attr_ig1 = np.transpose(attribution[1].squeeze().cpu().detach().numpy(), (1, 2, 0))  # 32,32,3
    attr_ig2 = np.transpose(attribution[2].squeeze().cpu().detach().numpy(), (1, 2, 0))  # 32,32,3
    attr_ig3 = np.transpose(attribution[3].squeeze().cpu().detach().numpy(), (1, 2, 0))  # 32,32,3

    original_image0 = np.transpose((images[0].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

    original_image1 = np.transpose((images[1].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

    original_image2 = np.transpose((images[2].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

    original_image3 = np.transpose((images[3].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
    _ = viz.visualize_image_attr(None, original_image0,
                                 method="original_image", title="Original Image1")
    _ = viz.visualize_image_attr(None, original_image1,
                                 method="original_image", title="Original Image2")
    _ = viz.visualize_image_attr(None, original_image2,
                                 method="original_image", title="Original Image3")
    _ = viz.visualize_image_attr(None, original_image3,
                                 method="original_image", title="Original Image4")

    _ = viz.visualize_image_attr(attr_ig0, original_image0, method="blended_heat_map", sign="all",
                                 show_colorbar=True, title="0Overlayed Integrated Gradients")
    _ = viz.visualize_image_attr(attr_ig1, original_image1, method="blended_heat_map", sign="all",
                                 show_colorbar=True, title="1Overlayed Integrated Gradients")
    _ = viz.visualize_image_attr(attr_ig2, original_image2, method="blended_heat_map", sign="all",
                                 show_colorbar=True, title="2Overlayed Integrated Gradients")
    _ = viz.visualize_image_attr(attr_ig3, original_image3, method="blended_heat_map", sign="all",
                                 show_colorbar=True, title="3Overlayed Integrated Gradients")


def get_distance(att1, att2):
    cos_sim_sum = 0.0
    n = len(att1)
    for i in range(n):
        att1_flat = att1[i].flatten()  # tensor 3072
        att2_flat = att2[i].flatten()  # tensor 3072

        # temp_cos_sim = torch.dot(att1_flat, att2_flat)
        # temp_cos_sim = torch.nn.functional.relu(temp_cos_sim, inplace=True)
        # cos_sim = (temp_cos_sim / (torch.norm(att1_flat) * torch.norm(att2_flat))).cpu().detach().numpy()
        # cos_sim_sum = cos_sim_sum + cos_sim

        cos_sim = (torch.dot(att1_flat, att2_flat) / (
                    torch.norm(att1_flat) * torch.norm(att2_flat))).cpu().detach().numpy()
        # cos_sim = 0.5+0.5*cos_sim
        cos_sim_sum = cos_sim_sum + cos_sim
    return n / cos_sim_sum


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
        # temp_cos_sim = torch.nn.functional.relu(temp_cos_sim, inplace=True)
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

        cos_sim = (temp_cos_sim / (torch.norm(att1_flat) * torch.norm(att2_flat))).cpu().detach().numpy()
        cos_sim_sum = cos_sim_sum + cos_sim
    # print(sum)
    # print(n / cos_sim_sum)
    # print(n / cos_sim_sum)

    if cos_sim_sum == 0:
        return 1
    else:
        return n / cos_sim_sum


def show_images_green(att, images):
    _ = viz.visualize_image_attr_multiple(np.transpose(att[0].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(images[0].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          show_colorbar=True,
                                          outlier_perc=2,
                                          )
    _ = viz.visualize_image_attr_multiple(np.transpose(att[1].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(images[1].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          show_colorbar=True,
                                          outlier_perc=2,
                                          )
    _ = viz.visualize_image_attr_multiple(np.transpose(att[2].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(images[2].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          show_colorbar=True,
                                          outlier_perc=2,
                                          )
    _ = viz.visualize_image_attr_multiple(np.transpose(att[3].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(images[3].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          show_colorbar=True,
                                          outlier_perc=2,
                                          )


def load_dataset(dataset, train, download=False):
    try:
        dataset_class = eval(f"torchvision.datasets.{dataset}")
    except:
        raise NotImplementedError(f"Dataset {dataset} is not implemented by pytorch.")

    if "MNIST" in dataset:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
    elif dataset == "CIFAR100":
        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        if train:
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                                            transforms.ToTensor(), normalize])
        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(), normalize])
        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])

    try:
        data = dataset_class(root='./data', train=train, download=download, transform=transform)
        # data.data = data.data[0:49000,:]
        # data.targets = data.targets[0:49000]
    except:
        if train:
            data = dataset_class(root='./data', split="train", download=download, transform=transform)
        else:
            data = dataset_class(root='./data', split="test", download=download, transform=transform)

    return data
