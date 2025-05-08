# -*- coding: utf-8 -*-
"""
Created on May 8 2025

@author: Omar Al-maqtari
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import gaussian_filter, laplace


def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue 
    g = gaussian_filter(n,sigma)
    gaussiankernel = torch.from_numpy(g)
    
    return gaussiankernel.float()

def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue
    l = laplace(n)
    laplacekernel = torch.from_numpy(l)
    
    return laplacekernel.float()


class SEM(nn.Module):
    def __init__(self, ch_out, reduction=4):
        super(SEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, reduction, kernel_size=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(reduction, ch_out, kernel_size=1,bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        
        return x * y.expand_as(x)
    
    
# Edge Extraction Attention
class EEA(nn.Module):
    def __init__(self, ch_in, kernel=3):
        super(EEA, self).__init__()
        
        self.groups = ch_in
        self.gk = gaussiankernel(ch_in, int(ch_in/ch_in), kernel, kernel-2, 0.9)
        self.lk = laplaceiankernel(ch_in, int(ch_in/ch_in), kernel, 0.9)
        self.gk = nn.Parameter(self.gk, requires_grad=False)
        self.lk = nn.Parameter(self.lk, requires_grad=False)
        
        self.q = nn.Sequential(
            nn.Conv2d(ch_in, ch_in//2, kernel_size=1, groups=4),
            nn.InstanceNorm2d(ch_in//2),
            nn.PReLU(num_parameters=ch_in//2, init=0.02)
            )
        self.k = nn.Sequential(
            nn.Conv2d(ch_in, ch_in//2, kernel_size=1, groups=4),
            nn.InstanceNorm2d(ch_in//2),
            nn.PReLU(num_parameters=ch_in//2, init=0.02)
            )
        self.v = nn.Sequential(
            nn.Conv2d(ch_in, ch_in//2, kernel_size=1, groups=4),
            nn.InstanceNorm2d(ch_in//2),
            nn.PReLU(num_parameters=ch_in//2, init=0.02),
            )
        self.conv = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_in//2, ch_in, kernel_size=1, groups=4),
            nn.GroupNorm(4, ch_in),
            nn.PReLU(num_parameters=ch_in, init=0.01)
            )
        self.sem1 = SEM(ch_in)
        self.sem2 = SEM(ch_in)
        self.softmax = nn.Softmax(-1)
        self.prelu = nn.PReLU(num_parameters=ch_in, init=0.01)
    
    def forward(self, x):
        b, _, h, w = x.size()
        
        DoG = F.conv2d(x, self.gk.to(x.device), padding='same',groups=self.groups)
        LoG = F.conv2d(DoG, self.lk.to(x.device), padding='same',groups=self.groups)
        q = self.q(DoG-x).view(b, -1, h*w)
        k = self.k(LoG).view(b, -1, h*w)
        v = self.v(LoG+(DoG-x)).view(b, -1, h*w)
        
        _, c, _ = k.shape
        q = torch.matmul(q, k.transpose(-1, -2)) * (c ** -0.5)
        q = self.softmax(q)
        v = torch.matmul(q, v).contiguous().view(b, c, h, w)
        
        v = self.conv(v)
        
        v = self.sem1(v) + v
        x = self.sem2(x) + x
        
        return self.prelu(x+v)