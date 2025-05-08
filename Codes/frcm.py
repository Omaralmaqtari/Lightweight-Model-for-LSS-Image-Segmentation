# -*- coding: utf-8 -*-
"""
Created on May 8 2025

@author: Omar Al-maqtari
"""
import torch
from torch import nn
import torch.nn.functional as F

class FRCM(nn.Module):
    def __init__(self,ch_ins, ch_out):
        super(FRCM,self).__init__()
        n_sides = len(ch_ins)
        
        self.reducers = nn.ModuleList([])
        for i in range(n_sides):
            self.reducers.append(nn.Conv2d(ch_ins[i], ch_out, kernel_size=1))
            
        self.gn = nn.GroupNorm(1, ch_out)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.05)
        
        self.fused = nn.Sequential(
                nn.Conv2d(ch_out*n_sides, ch_out, kernel_size=1),
                nn.PReLU(num_parameters=ch_out, init=0.05)
                )
            
    def forward_sides(self, sides, img_shape):
        late_sides = []
        for x, conv in zip(sides, self.reducers):
            x = self.prelu(self.gn(conv(x)))
            x = F.interpolate(x, size=img_shape, mode='bilinear', align_corners=True)
            late_sides.append(x)
            
        return late_sides
    
    def forward(self, sides, img_shape):
        late_sides = self.forward_sides(sides, img_shape)
        
        fused = self.fused(torch.cat(late_sides,1))
        late_sides.append(fused)
        
        return torch.cat(late_sides,1)
