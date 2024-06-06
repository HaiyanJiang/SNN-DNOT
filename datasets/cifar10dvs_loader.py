# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:42:14 2023

@author: Haiyan.Jiang
@email: Haiyan.Jiang @mbzuai.ac.ae
"""

# from torch.utils.data import DataLoader


import torch.utils.data as data
import torchvision.transforms as transforms
from .augmentation import ToPILImage, Resize, Padding, RandomCrop, ToTensor, Normalize, RandomHorizontalFlip
from .cifar10_dvs import CIFAR10DVS



# ## Change to your own data dir
DIR = {
    'CIFAR10': '/l/users/haiyan.jiang/datasets/CIFAR10',
    'CIFAR100': '/l/users/haiyan.jiang/datasets/CIFAR100',
    'CIFAR10DVS': '/l/users/haiyan.jiang/datasets/CIFAR10DVS',
    'CIFAR10_DVS': '/l/users/haiyan.jiang/datasets/CIFAR10_DVS',
    'MNIST': '/l/users/haiyan.jiang/datasets/',
    'ImageNet': '/l/users/haiyan.jiang/datasets/',
    'Tiny-ImageNet': '/l/users/haiyan.jiang/datasets/tiny-imagenet-200/'
}



# # ## Change to your own data dir
# DIR = {
#     'CIFAR10': './datasets/CIFAR10',
#     'CIFAR100': './datasets/CIFAR100',
#     'CIFAR10DVS': './datasets/CIFAR10DVS',
#     'CIFAR10_DVS': './datasets/CIFAR10_DVS',
#     'MNIST': './datasets/',
#     'ImageNet': './datasets/',
#     'Tiny-ImageNet': './datasets/tiny-imagenet-200/'
# }





# train_loader, test_loader = data_cifar10_dvs_loader(time_steps=args.T, batch_size=args.b, num_workers=args.j)


def data_cifar10_dvs_loader(time_steps, batch_size, num_workers):
    transform_train = transforms.Compose([
        ToPILImage(),
        Resize(48),
        Padding(4),
        RandomCrop(size=48, consistent=True),
        ToTensor(),
        Normalize((0.2728, 0.1295), (0.2225, 0.1290)),
    ])

    transform_test = transforms.Compose([
        ToPILImage(),
        Resize(48),
        ToTensor(),
        Normalize((0.2728, 0.1295), (0.2225, 0.1290)),
    ])

    trainset = CIFAR10DVS(
        DIR['CIFAR10_DVS'], train=True, use_frame=True, frames_num=time_steps,
        split_by='number', normalization=None, transform=transform_train)
    train_data_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = CIFAR10DVS(
        DIR['CIFAR10_DVS'], train=False, use_frame=True, frames_num=time_steps,
        split_by='number', normalization=None, transform=transform_test)
    test_data_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return train_data_loader, test_data_loader