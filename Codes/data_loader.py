# -*- coding: utf-8 -*-
"""
Created on May 8 2025

@author: Omar Al-maqtari
"""
import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image


class ImageFolder(Dataset):
    def __init__(self, cfg, mode, aug_prob):
        """Initializes image paths and preprocessing module."""
        self.root = cfg.dataset_path
        self.image_height = cfg.image_height
        self.image_width = cfg.image_width
        self.dataset = cfg.dataset
        self.mode = mode
        self.aug_prob = aug_prob
        
        self.image_paths = sorted(list(os.listdir(self.root+mode+'/')))
        self.GT_paths = sorted(list(os.listdir(self.root+mode+'_lab/')))
        print("image count in {} path: {}".format(self.mode, len(self.image_paths)))

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        GT_path = self.GT_paths[index]
        image = Image.open(self.root+self.mode+'/'+image_path).resize((self.image_width, self.image_width), resample=Image.BICUBIC)
        GT = Image.open(self.root+self.mode+'_lab/'+GT_path).resize((self.image_width, self.image_width), resample=Image.NEAREST)
        
        GT = np.asarray(GT).astype(np.int64)
        GT_values = np.unique(GT)
        if self.dataset == 'Axial':
            for i in range(len(GT_values)):
                GT[GT == GT_values[i]] = i
        elif self.dataset == "Sagittal":
            for i in range(len(GT_values)):
                GT[GT == GT_values[i]] = i
                
        image = T.ToTensor()(image).type(torch.float32)
        GT = T.ToTensor()(GT).type(torch.int64)
        
        if random.random() < self.aug_prob and self.mode == 'train':
            sharpness_factor = random.choice([2, 3, 4])
            Transform = T.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=0.25)
            image = Transform(image)
            
            if random.random() < 0.5:
                Transform = T.RandomRotation((90, 90))
                image = Transform(image)
                GT = Transform(GT)
                
            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)
                
            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)
            
        if image.shape[0] != 1 and image.shape[0] != 3:
            image = image.mean(dim=0, keepdim=True)
        if GT.shape[0] != 1:
            GT = T.Grayscale(num_output_channels=1)(GT)
            
        return image, GT, image_path


def get_loader(cfg, mode, aug_prob):
    """Builds and returns Dataloader."""
    
    dataset = ImageFolder(cfg, mode, aug_prob)
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False)
    
    return data_loader
