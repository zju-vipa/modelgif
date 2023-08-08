from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
from math import isnan
import numpy as np
import cv2
from PIL import Image
from copy import deepcopy
from tqdm import tqdm


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


def cut_mix(sample_num):
    # normal cutmix
    input_path = f"./data/reference_data_{sample_num}"
    output_path = f"./data/reference_data_cutmix_{sample_num}"

    # adv + cutmix
    # input_path = f"./data/reference_data_adv0_{sample_num}"
    # output_path = f"./data/reference_data_advcm_{sample_num}"

    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = OnlyImageDataset(input_path, transform=transform_test)
    BATCH_SIZE = 8
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)

    beta = 1

    random_seed = 3

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        """1.论文里的公式2，求出B的rw,rh"""
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)

        # uniform
        """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        #限制坐标区域不超过样本大小

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        """3.返回剪裁B区域的坐标值"""
        return bbx1, bby1, bbx2, bby2

    os.makedirs(output_path, exist_ok=True)
    for i, (input,
            image_name) in enumerate(tqdm(testloader, total=len(testloader))):
        # generate mixed sample
        imgs = deepcopy(input)
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        # print(bbx1, bby1, bbx2, bby2)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2,
                                                bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                   (input.size()[-1] * input.size()[-2]))

        imgs = imgs.transpose(1, 3).transpose(1, 2).numpy() * 255
        imgs[:, :, :, :] = imgs[:, :, :, ::-1]

        for image, im_name in zip(imgs, image_name):
            cv2.imwrite(os.path.join(output_path, im_name), image)


if __name__ == '__main__':
    for sn in ["1000"]: # 16, 32, 64, 128, 256, 512, 768, 1000, 2000, 3000
        cut_mix(sn)