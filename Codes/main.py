# -*- coding: utf-8 -*-
"""
Created on May 8 2025

@author: Omar Al-maqtari
"""
import os
import argparse
from trainer import Trainer
from torch.backends import cudnn
import torch

def main(cfg):
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if cfg.model_type not in ['M']:
        print('ERROR!! model_type should be selected in M')
        print('Your input for model_type was %s'%cfg.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
        cfg.result_path = os.path.join(cfg.result_path,cfg.model_type)
    
    print(cfg)
    
    trainer = Trainer(cfg)
    
    # Train and sample the images
    if cfg.mode == 'train':
        trainer.train()
    elif cfg.mode == 'test':
        trainer.test()
    
        
if __name__ == '__main__':
    for model_type in ['M']:
        for dataset in ['Sagittal']: # 'Axial_T1', 'Axial_T2', 'Axial_Composite', 'Sagittal'
            if dataset == 'Axial_T1' or dataset == 'Axial_T2' or dataset == 'Axial_Composite':
                img_in = 1
                ch_out = 5
                if dataset.find('T1') >= 0:
                    modality = 'T1'
                elif dataset.find('T2') >= 0:
                    modality = 'T2'
                elif dataset.find('Composite') >= 0:
                    modality = 'Composite'
                    img_in = 3
                dataset = 'Axial'
            elif dataset == 'Sagittal':
                modality = ''
                img_in = 3
                ch_out = 6
            for mode in ['train', 'test']: #'train', 'test'
                parser = argparse.ArgumentParser()
                if dataset == 'Axial':
                    class_weight = [5.5800, 12.3100, 62.6806, 35.1238, 0.2129] # Class_weights for the Axial_T1
                elif dataset == 'Sagittal':
                    class_weight = [0.2963, 0.7194, 1.9137, 2.6036, 4.8809, 8.0565] # Class_weights for Sagittal
                
                # hyperparameters
                parser.add_argument('--img_in', type=int, default=img_in)
                parser.add_argument('--ch_out', type=int, default=ch_out)
                parser.add_argument('--image_height', type=int, default=256)
                parser.add_argument('--image_width', type=int, default=256)
                parser.add_argument('--cuda', type=str, default='cuda:0')
                parser.add_argument('--lr', type=float, default=0.001)
                parser.add_argument('--epochs', type=int, default=150)
                parser.add_argument('--epochs_decay', type=int, default=4)
                parser.add_argument('--batch_size', type=int, default=16)
                parser.add_argument('--aug_prob', type=float, default=0.2)
                parser.add_argument('--class_weight', type=float, default=class_weight)
                parser.add_argument('--parameters', type=int, default=0)
                
                # datasets and training settings
                parser.add_argument('--mode', type=str, default=mode, help='train, test')
                parser.add_argument('--report_name', type=str, default=dataset +'_'+ modality +'_'+ model_type)
                parser.add_argument('--dataset', type=str, default=dataset)
                parser.add_argument('--modality', type=str, default=modality)
                parser.add_argument('--model_type', type=str, default=model_type, help='M')
                parser.add_argument('--optimizer_type', type=str, default='Adam', help='Adam, AdamW, RMSprop')
                parser.add_argument('--lr_sch_type', type=str, default='ReduceLROnPlateau', help='ReduceLROnPlateau, CosineAnnelingWarmupRestarts')
                parser.add_argument('--model_path', type=str, default='./weights/')
                parser.add_argument('--result_path', type=str, default='./results/')
                parser.add_argument('--dataset_path', type=str, default='./'+ dataset +'/'+ modality +'/')
                parser.add_argument('--SR_path', type=str, default='./'+ dataset +'/'+ modality +'/SR/')
                parser.add_argument('--wandb', type=bool, default=False, help='if wandb is ture, you need to set its inputs (i.e. entity, project, etc.) manually in the training file, default: False')
                cfg = parser.parse_args()
                main(cfg)        