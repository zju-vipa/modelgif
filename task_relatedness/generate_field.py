import os

import argparse

parser = argparse.ArgumentParser(description='PyTorch Image Training')
parser.add_argument('-r',
                    '--root',
                    default='./result_field_featuremap_tk',
                    help='save root')
parser.add_argument('-i',
                    '--image-path',
                    default="reference_data_1000",
                    help='image path')
parser.add_argument('-p', '--path', default='result_1000', help='result path')
parser.add_argument('-g',
                    '--gpu',
                    default=None,
                    type=str,
                    help='index of GPU to use.')
parser.add_argument('--grad', action="store_false", help='grad compare')
parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
parser.add_argument('-m',
                    '--method',
                    default=0,
                    type=int,
                    help='integral methods')
parser.add_argument('-n', '--nsteps', default=50, type=int, help='nsteps')
parser.add_argument('--exp',
                    default=1,
                    type=int,
                    help='different attribution targets')
parser.add_argument('--tk', default=1, type=int, help='topk')
parser.add_argument('--s256', action="store_true", help='use size 256')
parser.add_argument('--ig',
                    action="store_true",
                    help='use integrated gradient')
parser.add_argument('-b',
                    '--baseline',
                    default=-1,
                    type=int,
                    help='baseline for attribution')
args = parser.parse_args()
if args.gpu != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import visualpriors
import subprocess
from field_generator import FieldGenerator
import numpy as np
from captum.attr import InputXGradient, IntegratedGradients
import matplotlib.pyplot as plt
from tools.utils import decode_segmap, get_distance
# import pandas as pd
import numpy as np
from torchvision.utils import make_grid, save_image
import random
from random import Random
# class_object segment_semantic segment_unsup25d  segment_unsup2d  normal
feature_type = 'autoencoding'
label = 0
img_file_path = os.path.join('./data', args.image_path)
pth = os.path.join(args.root, args.path)
# list of 17 tasks
task_list = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d  \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_1000'

# 17 task
task_list_name = 'autoencoding curvature denoising edge_texture edge_occlusion \
keypoints2d keypoints3d \
reshading depth_zbuffer depth_euclidean normal \
room_layout segment_unsup25d segment_unsup2d vanishing_point \
segment_semantic class_object'

task_list = task_list_name.split()
list_zero = [0 for i in range(17)]
task_dict = dict(zip(task_list, list_zero))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def shuffle_and_check_difference(li: list):
    raw_li = li
    li = li.copy()
    rd = Random(args.seed)
    rd.shuffle(li)
    while any(filter(lambda x: x[0] == x[1], zip(raw_li, li))):
        rd.shuffle(li)
    return li


img_file_list = os.listdir(img_file_path)
img_file_baseline = shuffle_and_check_difference(img_file_list)
methods = [
    'gausslegendre', 'riemann_right', 'riemann_left', 'riemann_middle',
    'riemann_trapezoid'
]
for key in task_dict:
    if os.path.exists(f"{pth}/{key}/{key}_att.npy"):
        print(f"exist {pth}/{key}/{key}_att.npy")
        continue
    setup_seed(args.seed)
    print(key)
    feature_type = key
    model = visualpriors.get_nets(feature_type, device='cpu')
    if hasattr(model, "encoder"):
        model = model.encoder
    model.eval()
    model.cuda()

    image = Image.open(
        r'./data/reference_data_1000/point_0_view_1_domain_rgb.png')
    img = TF.to_tensor(TF.resize(image, 128)) * 2 - 1
    img = img.unsqueeze_(0)
    img.requires_grad = True
    img = img.cuda()
    out = model(img)
    class_num = out.shape[1]
    attss = []

    def process_image(img_file_name):
        img_path = os.path.join(img_file_path, img_file_name)
        image = Image.open(img_path)
        if args.exp == 2 or args.s256:
            img_size = 256
        else:
            img_size = 128  # 128
        img = TF.to_tensor(TF.resize(image, img_size)) * 2 - 1  # 之前都是256的
        img = img.unsqueeze_(0)
        return img

    if class_num > 3:

        for idx, (img_name, baseline_name) in enumerate(
                zip(img_file_list, img_file_baseline)):
            print(pth, key, ":", idx, "/", len(img_file_list))
            img = process_image(img_name).cuda()  # 1,3,128,128
            img.requires_grad = True

            model = visualpriors.get_nets(feature_type, device='cpu')
            if hasattr(model, "encoder"):
                model = model.encoder
            model.eval()
            model = model.cuda()

            output = model(img)
            # out_max
            out_max = torch.argmax(output, dim=1, keepdim=True)
            selected_inds = torch.zeros_like(output).scatter_(
                1, out_max, 1)  # 17个类的one-hot编码
            if args.exp == 0:
                # attribute as segmentation task
                def agg_segmentation_wrapper(inp):

                    model_out = model(
                        inp
                    )  # inp:tensor 1,3,256,256 model_out:tensor 1,17,256,256
                    return (model_out * selected_inds).sum(dim=(2, 3))

                for i in range(class_num):
                    label = i
                    if not args.ig:
                        input_x_gradient = FieldGenerator(
                            agg_segmentation_wrapper, if_cumsum=args.grad)
                    else:
                        input_x_gradient = IntegratedGradients(
                            agg_segmentation_wrapper)
                    if args.baseline == 0:
                        att = input_x_gradient.attribute(
                            img,
                            target=i,
                            n_steps=args.nsteps,
                            method=methods[args.method])  # tensor 1,50,X
                    else:
                        baseline = process_image(baseline_name).cuda()
                        att = input_x_gradient.attribute(
                            img,
                            target=i,
                            n_steps=args.nsteps,
                            baselines=baseline,
                            method=methods[args.method])  # tensor 1,50,X
                    if i == 0:
                        atts = att
                    else:
                        atts = att + atts
                if not args.ig:
                    atts = atts.mean(2).flatten(2)
                else:
                    atts = atts.mean(1)
            elif args.exp == 1:
                flat_out = output.view(output.shape[0], output.shape[1], -1)
                _, out_max = torch.topk(flat_out, k=args.tk, dim=2)
                selected_inds = torch.zeros_like(flat_out).scatter_(
                    2, out_max, 1).reshape_as(output)

                def agg_segmentation_wrapper(inp):

                    model_out = model(inp)
                    return (model_out * selected_inds).sum(dim=(1, 2, 3))

                if not args.ig:
                    input_x_gradient = FieldGenerator(agg_segmentation_wrapper,
                                                      if_cumsum=args.grad)
                else:
                    input_x_gradient = IntegratedGradients(
                        agg_segmentation_wrapper)
                if args.baseline == 0:
                    atts = input_x_gradient.attribute(
                        img, n_steps=args.nsteps,
                        method=methods[args.method])  # tensor 1,50,X
                else:
                    baseline = process_image(baseline_name).cuda()
                    atts = input_x_gradient.attribute(
                        img,
                        n_steps=args.nsteps,
                        baselines=baseline,
                        method=methods[args.method])  # tensor 1,50,X

                if not args.ig:
                    atts = atts.mean(2).flatten(2)
                else:
                    atts = atts.mean(1)
            elif args.exp == 2:

                def agg_segmentation_wrapper(inp):

                    model_out = model(
                        inp
                    )  # inp:tensor 1,3,256,256 model_out:tensor 1,17,256,256
                    return (model_out * selected_inds).sum(dim=(2, 3))

                for i in range(class_num):
                    label = i
                    if not args.ig:
                        input_x_gradient = FieldGenerator(
                            agg_segmentation_wrapper, if_cumsum=args.grad)
                    else:
                        input_x_gradient = IntegratedGradients(
                            agg_segmentation_wrapper)
                    if args.baseline == 0:
                        att = input_x_gradient.attribute(
                            img,
                            target=i,
                            n_steps=args.nsteps,
                            method=methods[args.method])  # tensor 1,50,X
                    else:
                        baseline = process_image(baseline_name).cuda()
                        att = input_x_gradient.attribute(
                            img,
                            target=i,
                            n_steps=args.nsteps,
                            baselines=baseline,
                            method=methods[args.method])  # tensor 1,50,X
                    if i == 0:
                        atts = att
                    else:
                        atts = att + atts

                if args.ig:
                    atts = TF.resize(atts,
                                     128,
                                     interpolation=InterpolationMode.NEAREST)
                else:
                    atts = TF.resize(
                        atts.view(-1, 256, 256),
                        128,
                        interpolation=InterpolationMode.NEAREST).reshape(
                            *(*atts.shape[:-3], -1))

            elif args.exp == 3:
                def agg_segmentation_wrapper(inp):

                    model_out = model(
                        inp
                    ) 
                    return model_out.sum(dim=(2, 3))

                for i in range(class_num):
                    label = i
                    if not args.ig:
                        input_x_gradient = FieldGenerator(
                            agg_segmentation_wrapper, if_cumsum=args.grad)
                    else:
                        input_x_gradient = IntegratedGradients(
                            agg_segmentation_wrapper)
                    if args.baseline == 0:
                        att = input_x_gradient.attribute(
                            img,
                            target=i,
                            n_steps=args.nsteps,
                            method=methods[args.method])  # tensor 1,50,X
                    else:
                        baseline = process_image(baseline_name).cuda()
                        att = input_x_gradient.attribute(
                            img,
                            target=i,
                            n_steps=args.nsteps,
                            baselines=baseline,
                            method=methods[args.method])  # tensor 1,50,X
                    if i == 0:
                        atts = att
                    else:
                        atts = att + atts
                if not args.ig:
                    atts = atts.mean(2).flatten(2)
                else:
                    atts = atts.mean(1)
            elif args.exp == 4:
                flat_out = output.view(output.shape[0], -1)
                _, out_max = torch.topk(flat_out, k=1, dim=1)
                selected_inds = torch.zeros_like(flat_out).scatter_(
                    1, out_max, 1).reshape_as(output)  

                def agg_segmentation_wrapper(inp):

                    model_out = model(
                        inp
                    )
                    return (model_out * selected_inds).sum(dim=(1, 2, 3))

                if not args.ig:
                    input_x_gradient = FieldGenerator(agg_segmentation_wrapper,
                                                      if_cumsum=args.grad)
                else:
                    input_x_gradient = IntegratedGradients(
                        agg_segmentation_wrapper)
                if args.baseline == 0:
                    atts = input_x_gradient.attribute(
                        img, n_steps=args.nsteps,
                        method=methods[args.method])  # tensor 1,50,X
                else:
                    baseline = process_image(baseline_name).cuda()
                    atts = input_x_gradient.attribute(
                        img,
                        n_steps=args.nsteps,
                        baselines=baseline,
                        method=methods[args.method])  # tensor 1,50,X
                if not args.ig:
                    atts = atts.mean(2).flatten(2)
                else:
                    atts = atts.mean(1)
            elif args.exp == 5:

                def agg_segmentation_wrapper(inp):

                    model_out = model(
                        inp
                    )  # inp:tensor 1,3,256,256 model_out:tensor 1,17,256,256
                    return (model_out * selected_inds).sum(dim=(1, 2, 3))

                if not args.ig:
                    input_x_gradient = FieldGenerator(agg_segmentation_wrapper,
                                                      if_cumsum=args.grad)
                else:
                    input_x_gradient = IntegratedGradients(
                        agg_segmentation_wrapper)
                if args.baseline == 0:
                    atts = input_x_gradient.attribute(
                        img, n_steps=args.nsteps,
                        method=methods[args.method])  # tensor 1,50,X
                else:
                    baseline = process_image(baseline_name).cuda()
                    atts = input_x_gradient.attribute(
                        img,
                        n_steps=args.nsteps,
                        baselines=baseline,
                        method=methods[args.method])  # tensor 1,50,X

                if not args.ig:
                    atts = atts.mean(2).flatten(2)
                else:
                    atts = atts.mean(1)
            elif args.exp == 6:

                def agg_segmentation_wrapper(inp):

                    model_out = model(
                        inp
                    )  # inp:tensor 1,3,256,256 model_out:tensor 1,17,256,256
                    return model_out.sum(dim=(1, 2, 3))

                if not args.ig:
                    input_x_gradient = FieldGenerator(agg_segmentation_wrapper,
                                                      if_cumsum=args.grad)
                else:
                    input_x_gradient = IntegratedGradients(
                        agg_segmentation_wrapper)
                if args.baseline == 0:
                    atts = input_x_gradient.attribute(
                        img, n_steps=args.nsteps,
                        method=methods[args.method])  # tensor 1,50,X
                else:
                    baseline = process_image(baseline_name).cuda()
                    atts = input_x_gradient.attribute(
                        img,
                        n_steps=args.nsteps,
                        baselines=baseline,
                        method=methods[args.method])  # tensor 1,50,X

                if not args.ig:
                    atts = atts.mean(2).flatten(2)
                else:
                    atts = atts.mean(1)
            else:
                raise NotImplementedError
            attss.append(atts.cpu().float().detach())
    else:
        print(f"skip {feature_type}")
        continue
    att_cpu = torch.cat(attss, 0).numpy()
    print(att_cpu.shape)
    if not os.path.exists(f"{pth}/{key}"):
        os.makedirs(f"{pth}/{key}", exist_ok=True)
    np.save(f"{pth}/{key}/{key}_att.npy", att_cpu)

    print("{} finished".format(key))