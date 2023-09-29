import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from math import isnan
import numpy as np
import cv2
from PIL import Image
from copy import deepcopy
import random
from torch import nn
from tqdm import tqdm
import visualpriors


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class OnlyImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        super(OnlyImageDataset, self).__init__()
        image_name_list = os.listdir(image_dir)
        self.images = []
        self.images_name = []
        for image_name in image_name_list:
            image_p = os.path.join(image_dir, image_name)
            self.images.append(Image.open(image_p))
            self.images_name.append(image_name)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(self.images[item]), self.images_name[item]


def pgd(model, data, labels, random_start=False):  # adv0
    # one adv
    # flat_labels = labels.view(labels.shape[0], labels.shape[1], -1)
    # _, labels_max = torch.topk(flat_labels, k=1, dim=2)
    # labels = torch.zeros_like(flat_labels).scatter_(2, labels_max,
    #                                                 1).reshape_as(labels)
    #
    data = data.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    epsilon = 16 / 255
    k = 8
    a = epsilon / max(k, 1e-8)
    data_max = data + epsilon
    data_min = data - epsilon
    d_min = 0
    d_max = 1
    data_max.clamp_(d_min, d_max)
    data_min.clamp_(d_min, d_max)
    perturbed_data = data.clone().detach()
    if random_start:
        # Starting at a uniformly random point
        perturbed_data = perturbed_data + torch.empty_like(
            perturbed_data).uniform_(-1 * epsilon, epsilon)
        perturbed_data = torch.clamp(perturbed_data, min=0, max=1).detach()
    for _ in range(k):
        perturbed_data.requires_grad = True
        outputs = model(perturbed_data * 2 - 1)
        loss = nn.L1Loss()
        cost = -1 * loss(outputs, labels)
        # cost = (outputs * labels).mean()
        # print(cost)
        # Update adversarial images
        cost.backward()
        gradient = perturbed_data.grad.clone().cuda()
        perturbed_data.grad.zero_()
        with torch.no_grad():
            perturbed_data.data -= a * torch.sign(gradient)
            perturbed_data.data = torch.max(
                torch.min(perturbed_data, data_max), data_min)
    return perturbed_data.cpu().detach()


task_list_name = 'autoencoding curvature denoising edge_texture edge_occlusion \
keypoints2d keypoints3d \
reshading depth_zbuffer depth_euclidean normal \
segment_unsup25d segment_unsup2d \
segment_semantic'

task_list = task_list_name.split()


def adv(sample_num):
    setup_seed(0)
    input_path = f"./data/reference_data_{sample_num}"
    # output_path = f"./data/reference_data_adv1_{sample_num}"
    output_path = f"./data/reference_data_adv0_{sample_num}"
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = OnlyImageDataset(input_path, transform=transform_test)
    BATCH_SIZE = 4
    testloader = torch.utils.data.DataLoader(testset,
                                             num_workers=8,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)
    os.makedirs(output_path, exist_ok=True)
    for i, (input,
            image_name) in enumerate(tqdm(testloader, total=len(testloader))):
        # generate mixed sample
        imgs = deepcopy(input)
        imgs = imgs.cuda()
        # adv
        feature_type = task_list[i % len(task_list)]
        model = visualpriors.get_nets(feature_type, device='cpu')
        model.eval()
        model = model.cuda()
        output = model.encoder(imgs * 2 - 1)
        imgs = pgd(model.encoder, imgs, output)  # adv0
        imgs = imgs.transpose(1, 3).transpose(1, 2).numpy() * 255
        imgs[:, :, :, :] = imgs[:, :, :, ::-1]
        for image, img_name in zip(imgs, image_name):
            cv2.imwrite(os.path.join(output_path, img_name), image)
    # labels = torch.cat(labels,dim=0)


if __name__ == '__main__':
    for sn in ["1000"]: 
        adv(sn)
