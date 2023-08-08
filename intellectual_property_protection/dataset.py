import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image


def normalize(image):
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = np.array(image_data)
    img_copy = torch.zeros(image.shape)
    for i in range(3):
        img_copy[i, :, :] = (image[i, :, :] - image_data[0, i]) / image_data[1,
                                                                             i]

    return img_copy


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_CIFAR100 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.53561753, 0.48983628, 0.42546818),
                         (0.26656017, 0.26091456, 0.27394977))
])

transform_CIFAR10C_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4645897160947712, 0.6514782475490196, 0.5637088950163399),
        (0.18422159112571024, 0.3151505122530825, 0.26127269383599344))
])
transform_CIFAR10C_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4645897160947712, 0.6514782475490196, 0.5637088950163399),
        (0.18422159112571024, 0.3151505122530825, 0.26127269383599344))
])


class dataset(Dataset):

    def __init__(self, name, train=False):
        super(dataset, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :] * 255, dtype='uint8')
        image = transform_train(image)
        return [image, label]


class dataset1(Dataset):

    def __init__(self, name, train=False):
        super(dataset1, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :] * 255, dtype='uint8')
        image = transform_test(image)

        return [image, label]


class dataset_field_sample(Dataset):

    def __init__(self, name, modes=[]):
        super(dataset_field_sample, self).__init__()

        self.name = name
        self.data = h5py.File(os.path.join("data", name), 'r')

        self.images = torch.tensor(self.data['/data'])
        self.labels = torch.tensor(self.data['/label'])
        self.preds = torch.tensor(self.data['/preds'])
        self.student_preds = torch.tensor(self.data['/student_preds'])
        self.raw_distance = torch.tensor(self.data['/distance'])
        if "min" in modes:
            self.distance_sur, _ = torch.min(self.raw_distance[:, :6], dim=1)
            self.distance_irr, _ = torch.min(self.raw_distance[:, 6:], dim=1)
        elif "max" in modes:
            self.distance_sur, _ = torch.max(self.raw_distance[:, :6], dim=1)
            self.distance_irr, _ = torch.max(self.raw_distance[:, 6:], dim=1)
        else:
            self.distance_sur = torch.mean(self.raw_distance[:, :6], dim=1)
            self.distance_irr = torch.mean(self.raw_distance[:, 6:], dim=1)
        self.trans = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                          (0.2023, 0.1994, 0.2010))
        li = [
            "distance_sur", "distance_irr", "images", "labels", "preds",
            "student_preds", "raw_distance"
        ]

        if "sort_by_gap" in modes:
            _, idx = torch.sort(self.distance_irr - self.distance_sur,
                                descending=True)
            for name in li:
                getattr(self, name)[:] = getattr(self, name)[idx]

        if "sort_by_distance_irr" in modes:
            _, idx = torch.sort(self.distance_irr, descending=True)
            for name in li:
                getattr(self, name)[:] = getattr(self, name)[idx]

        if "sort_by_distance_sur" in modes:
            _, idx = torch.sort(self.distance_sur)
            for name in li:
                getattr(self, name)[:] = getattr(self, name)[idx]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):

        label = self.labels[item]
        image = self.images[item]
        image = self.trans(image)

        return [image, label]


class dataset3(Dataset):

    def __init__(self, name, train=False):
        super(dataset3, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :] * 255, dtype='uint8')
        image = transform_CIFAR100(image)

        return [image, label]


class dataset4(Dataset):

    def __init__(self, name, train=False):
        super(dataset4, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])
        if train == True:
            self.transform = transform_CIFAR10C_train
        elif train == False:
            self.transform = transform_CIFAR10C_test

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :] * 255, dtype='uint8')
        image = self.transform(image)

        return [image, label]


class dataset5(Dataset):

    def __init__(self, name, train=False):
        super(dataset5, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):

        image = (np.squeeze(self.images[item, :, :, :]))
        image = np.transpose(image, [2, 0, 1])
        image = torch.tensor(image)
        image = normalize(image)
        label = torch.tensor(self.labels[item])
        return [image, label]


class dataset6(Dataset):

    def __init__(self, name, train=False):
        super(dataset6, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):

        image = (np.squeeze(self.images[item, :, :, :]))
        image = torch.tensor(image)
        image = normalize(image)
        label = torch.tensor(self.labels[item])

        return [image, label]


class dataset7(Dataset):

    def __init__(self, name, train=False):
        super(dataset7, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data", name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):

        image = (np.squeeze(self.images[item, :, :, :]))
        image = torch.tensor(image)
        label = torch.tensor(self.labels[item])
        return [image, label]


if __name__ == "__main__":
    ds = dataset_field_sample("dna_sample.h5", modes=["sort_by_gap"])

    print(ds.distance_sur[:50], len(ds))
    print(ds.distance_irr[:50])
