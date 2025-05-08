# -*- coding: utf-8 -*-
"""
Created on May 8 2025

@author: Omar Al-maqtari
"""
from model import M
import torch
from ptflops import get_model_complexity_info
import re

#Model thats already available
def flops(img_in, ch_out, img_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ch_in = [64, 128, 128, 256]
    model = M(img_in, ch_in, ch_out).to(device)
    
    macs, params = get_model_complexity_info(model, (img_in, img_size, img_size), as_strings=True, print_per_layer_stat=False, verbose=False)
    # Extract the numerical value
    flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    # Extract the unit
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    print('Computational  complexity at image size: ({},{})'.format(img_size,img_size))
    print('Computational complexity: {:<8}'.format(macs))
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))
    
    
if __name__ == '__main__':
    flops(1,5,256)
