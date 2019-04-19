# Reference Paper: Prototypical Networks for Few shot Learning in PyTorch
# Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf
# Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel

# Reference Code: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
# Reference Code Author: Daniele E. Ciriello

import os, torch
import numpy as np

from omniglot_dataset import OmniglotDataset
from mini_imagenet_dataset import MiniImageNet
from batch_sampler import BatchSampler

import torchvision.transforms as transforms
import torchvision.datasets as datasets

class DataLoader(object):

    def __init__(self, mode, arg_settings):
        self.mode = mode
        self.arg_settings = arg_settings
        self.data_loader = self.set_data_loader()

    def data_loader(self):
        return self.data_loader

    def set_data_loader(self):

        if self.arg_settings.data=='omniglot':
            dataset = OmniglotDataset(mode=self.mode, root=self.arg_settings.dataset_root+'/omniglot/')
            num_classes = len(np.unique(dataset.y))
            sampler = self.create_sampler(dataset.y)

        elif self.arg_settings.data=='imagenet':
            dataset = MiniImageNet(mode=self.mode, root=self.arg_settings.dataset_root+'/imagenet/')
            num_classes = len(np.unique(dataset.label))
            sampler = self.create_sampler(dataset.label)
            print('num_classes: ', num_classes,' for ',self.mode)
            #self.arg_settings.classes_per_it_tr = 60
            #self.arg_settings.classes_per_it_val = 16
            # Why these values were fixed to be 60 and 16?

        # check if number of classes in dataset is sufficient
        if self.mode == 'train':
            if (num_classes < self.arg_settings.classes_per_it_tr):
                raise(Exception('There are not enough classes in the dataset in order ' +
                                'to satisfy the chosen classes_per_it. Decrease the ' +
                                'classes_per_it_tr option and try again.'))
        else:
            if num_classes < self.arg_settings.classes_per_it_val:
                raise(Exception('There are not enough classes in the dataset in order ' +
                                'to satisfy the chosen classes_per_it. Decrease the ' +
                                'classes_per_it_val option and try again.'))

        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
        return dataloader

    def create_sampler(self, labels):
        if self.mode=='train':
            classes_per_it = self.arg_settings.classes_per_it_tr
            num_samples = self.arg_settings.num_support_tr + self.arg_settings.num_query_tr
        else:
            classes_per_it = self.arg_settings.classes_per_it_val
            num_samples = self.arg_settings.num_support_val + self.arg_settings.num_query_val

        return BatchSampler(labels=labels,
                            classes_per_it=classes_per_it,
                            num_samples=num_samples,
                            iterations=self.arg_settings.iterations)
