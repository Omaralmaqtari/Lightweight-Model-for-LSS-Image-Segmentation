# -*- coding: utf-8 -*-
"""
Created on May 8 2025

@author: Omar Al-maqtari
"""
import os
import wandb

from model import M
from deepLabv3 import DeepLabv3plus
from vit_seg_modeling import CONFIGS, SegViTv2
from swin_transformer import SwinTransformerSeg
from monai.networks.nets import SegResNet, UNETR, SwinUNETR, AttentionUnet

from torch import optim
from scheduler import CosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt


class initialize_model(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.lr_sch = None
        
    def get_model(self, model_type):
        if model_type == 'M':
            ch_in = [64, 128, 128, 256]
            self.model = M(self.cfg.img_in, ch_in, self.cfg.ch_out)
            
        elif model_type == 'SegViTv2':
            self.model = SegViTv2(self.cfg.img_in, CONFIGS['R50-ViT-B_16'], img_size=self.cfg.image_height, num_classes=self.cfg.ch_out)
            
        elif model_type == 'SwinTransformerSeg':
            self.model = SwinTransformerSeg(self.cfg.img_in, self.cfg.ch_out)
            
        elif model_type == 'SegResNet':
            self.model = SegResNet(spatial_dims=2, init_filters=32, in_channels=self.cfg.img_in, out_channels=self.cfg.ch_out,
                                   dropout_prob=0.2, num_groups=8, norm_name='GROUP', upsample_mode='deconv')
            
        elif model_type == 'UNETR':
            self.model = UNETR(in_channels=self.cfg.img_in, out_channels=self.cfg.ch_out, img_size=(self.cfg.image_height, self.cfg.image_height),
                               feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, norm_name='instance', 
                               conv_block=True, res_block=True, dropout_rate=0.0, spatial_dims=2, qkv_bias=False)
            
        elif model_type == 'SwinUNETR':
            self.model = SwinUNETR(img_size=(self.cfg.image_height, self.cfg.image_height), in_channels=self.cfg.img_in, out_channels=self.cfg.ch_out,
                                   depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), feature_size=24, norm_name='instance',
                                   drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0, normalize=True,
                                   use_checkpoint=False, spatial_dims=2, downsample='merging')
            
        elif model_type == 'AttentionUnet':
            self.model = AttentionUnet(spatial_dims=2, in_channels=self.cfg.img_in, out_channels=self.cfg.ch_out, channels=[64, 128, 256, 320],
                                       strides=[1, 1, 1, 1], kernel_size=3, up_kernel_size=3, dropout=0.0)
            
        elif model_type == 'DeepLabv3plus':
            self.model = DeepLabv3plus(self.cfg.img_in, self.cfg.ch_out)
            
        return self.model.to(self.cfg.cuda)
    
    def get_optimizer(self, optimizer_type, lr_sch_type):
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), self.cfg.lr)
        elif optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), self.cfg.lr, weight_decay=2e-4)
            
        if lr_sch_type == 'ReduceLROnPlateau':
            self.lr_sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.9, patience=self.cfg.epochs_decay)
        elif lr_sch_type == 'CosineAnnelingWarmupRestarts':
            self.lr_sch = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=self.cfg.epochs//2, 
                                                        max_lr=self.cfg.lr, min_lr=0.0001, warmup_steps=int(self.cfg.epochs*0.05))
            
        return self.optimizer, self.lr_sch


def wandbMetrics(dataset, experiment, lr, epoch, loss, IoU, Dc):
    class_labels = {
        'Axial': ['ivd', 'pe', 'ts', 'aap', 'bg'],
        'Sagittal': ['PosteriorA', 'PosteriorB', 'Vertebrae', 'IVD', 'Sacrum', 'bg']
        }
    
    labels = class_labels.get(dataset, [])
    Dc_scores = {f'Dc_{label}': round(Dc[i].item(), 4) for i, label in enumerate(labels)}
    IoU_scores = {f'IoU_{label}': round(IoU[i].item(), 4) for i, label in enumerate(labels)}
    
    experiment.log({
        'epoch': epoch,
        'learning_rate': lr,
        'loss': loss,
        'mean_IoU': round(IoU[-1].item(), 4),
        'mean_Dc': round(Dc[-1].item(), 4),
        **Dc_scores,
        **IoU_scores
        })
    
    
def wandbBestMertics(dataset, IoU, Dc):
    class_labels = {
        'Axial': ['ivd', 'pe', 'ts', 'aap', 'bg'],
        'Sagittal': ['PosteriorA', 'PosteriorB', 'Vertebrae', 'IVD', 'Sacrum', 'bg']
        }
    
    labels = class_labels.get(dataset, [])
    IoU_scores = {f'IoU_{label}': round(IoU[i].item(), 4) for i, label in enumerate(labels)}
    Dc_scores = {f'Dc_{label}': round(Dc[i].item(), 4) for i, label in enumerate(labels)}
    
    wandb.summary['mean_IoU.max'] = round(IoU[-1].item(), 4)
    for label, value in IoU_scores.items():
        wandb.summary[label] = value
    wandb.summary['mean_Dc.max'] = round(Dc[-1].item(), 4)
    for label, value in Dc_scores.items():
        wandb.summary[label] = value
        
        
def wandbTrainedModel(model_path, cfg):
    trained_model_artifact = wandb.Artifact(cfg.model_type, type="model", description=os.path.basename(model_path),
                metadata=dict(cfg))
    
    trained_model_artifact.add_file(model_path)
    wandb.log_artifact(trained_model_artifact)
    
    
def displayfigures(results, result_path, report_name):
    for i in range(len(results)):
        plt.Figure()
        plt.plot(results[i][1], marker='o', markersize=3, label="Train "+results[i][0])
        plt.plot(results[i][2], marker='s', markersize=3, label="Val "+results[i][0])
        plt.legend(loc="lower right")
        plt.xlabel("Epochs")
        plt.ylabel(results[i][0]+"%")
        if results[i][0] != "Loss":
            plt.ylim(0,100)
        plt.savefig(result_path+report_name+'_'+results[i][0]+'_results.png')
        plt.close()
    