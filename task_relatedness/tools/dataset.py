import numpy as np
import os
import torch
import torch.optim as optim
# from tools import utils
import os
import numpy as np
import torch
import copy
import torchvision
import torchvision.transforms as transforms
from tools.utils_zest import mean_std_to_array
from lime.wrappers.scikit_image import SegmentationAlgorithm

# 下载、加载数据集
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
            # transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
            #                                 transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
            #                                 transforms.ToTensor(), normalize])
            transform = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.ToTensor(), normalize])
        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            # transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
            #                                 transforms.ToTensor(), normalize])
            transform = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.ToTensor(), normalize])
        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])

    try:
        data = dataset_class(root='./data', train=train, download=download, transform=transform)
    except:
        if train:
            data = dataset_class(root='./data', split="train", download=download, transform=transform)
        else:
            data = dataset_class(root='./data', split="test", download=download, transform=transform)

    return data


# 获取数据集
def get_dataset(batch_size=1000, dataset="CIFAR100", trainset=None):
    if not os.path.exists("data"):
        os.mkdir("data")
    if trainset is None:
        trainset = load_dataset(dataset, True, download=True)
    else:
        trainset = trainset
    testset = load_dataset(dataset, False, download=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=True, num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=0, pin_memory=True)
    return trainset, testset


# 准备128张参考图片
def prepare_ref_data(save_name, dataset=None, data_size=128, ref_data=None, mean=None, std=None):
    if os.path.exists(f"data/{save_name}/ref_data.npy"):
        print("加载了哈")
        return np.load(f"data/{save_name}/ref_data.npy"), np.load(f"data/{save_name}/ref_label.npy")
    else:
        if not os.path.exists(f"data/{save_name}"):
            os.mkdir(f"data/{save_name}")
        if ref_data is None:
            assert dataset is not None
            ref_data = dataset.data[:data_size] # ndarray 128,32,32,3
            ref_label = dataset.targets[:data_size] # list 128
            if hasattr(dataset, 'transform'):
                for transform in dataset.transform.transforms:
                    if isinstance(transform, torchvision.transforms.transforms.Normalize):
                        mean, std = mean_std_to_array(transform.mean, transform.std, ref_data.shape[-1] == 3)
                        ref_data = (ref_data / 255 - mean) / std
        else:
            if mean is not None and std is not None:
                mean, std = mean_std_to_array(mean, std, ref_data.shape[-1] == 3)
                ref_data = (ref_data / 255 - mean) / std


        np.save(f"data/{save_name}/ref_data.npy", ref_data)
        np.save(f"data/{save_name}/ref_label.npy", ref_label)
        return ref_data, ref_label  # ref_data: ndarray 128,32,32,3    ref_label: list 128


# 准备lime的超级像素的切割块
def prepare_lime_segment(save_name, ref_data=None, dataset=None, mean=None, std=None):
    # if os.path.exists(f"data/{save_name}/segment.npy"):
    #     return np.load(f"data/{save_name}/segment.npy")
    # else:
        if not os.path.exists(f"data/{save_name}"):
            os.mkdir(f"data/{save_name}")
        if dataset is not None:
            if hasattr(dataset, 'transform'):
                for transform in dataset.transform.transforms:
                    if isinstance(transform, torchvision.transforms.transforms.Normalize):
                        mean, std = mean_std_to_array(transform.mean, transform.std, ref_data.shape[-1] == 3)
                        ref_data = (ref_data * std + mean) * 255
        elif mean is not None and std is not None:
            mean, std = mean_std_to_array(mean, std, ref_data.shape[-1] == 3)
            ref_data = (ref_data * std + mean) * 255  # [0,255]
        temp = []
        if ref_data.shape[1] == 3:
            ref_data = np.moveaxis(ref_data, 1, -1)  # ndarray 128,32,32,3
        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, ratio=0.2, max_dist=200)  # 快速移位图像分割算法
        for image in ref_data: # ndarray 32,32,3
            temp.append(segmentation_fn(image))  # 32*32 分簇\
            temp.append(segmentation_fn(image))  # 32*32 分簇
            tempa = segmentation_fn(image)  # ndarray 100,100
            # result = mark_boundaries(image, tempa)  # 标记边界
            # cv2.imshow("result", result)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        lime_segment = np.stack(temp)  # 128*32*32  128张参考图片，每张图片是32*32，用0-？进行标记，标记分割的区域。
        np.save(f"data/{save_name}/segment.npy", lime_segment)
        return lime_segment