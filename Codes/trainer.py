# -*- coding: utf-8 -*-
"""
Created on May 8 2025

@author: Omar Al-maqtari
"""
import os
import time
import wandb
from tqdm import tqdm
from datetime import datetime
from data_loader import get_loader

import torch

from model import Learner
from utils import initialize_model, wandbMetrics, wandbBestMertics, wandbTrainedModel, displayfigures

import csv
from evaluation import Evaluation
from monai.losses import DiceLoss


class Trainer(object):
    def __init__(self, cfg):
        # Config
        self.cfg = cfg
        self.wandb = cfg.wandb
        self.device = torch.device(cfg.cuda if torch.cuda.is_available() else 'cpu')

        # Paths
        self.result_path = cfg.result_path
        self.SR_path = cfg.SR_path
        self.model_path = os.path.join(cfg.model_path, cfg.report_name + '.pth')
        
        # Report file
        self.report_name = cfg.report_name
        self.report = open(self.result_path + self.report_name + '.txt', 'a+')
        self.report.write('\n' + str(datetime.now()))
        self.report.write('\n' + str(cfg))

        # Data loader
        self.train_loader = get_loader(cfg, mode='train', aug_prob=cfg.aug_prob)
        self.valid_loader = get_loader(cfg, mode='valid', aug_prob=cfg.aug_prob)
        self.test_loader = get_loader(cfg, mode='test', aug_prob=0.)
        
        # Hyper-parameters
        self.lr = cfg.lr
        self.aug_prob = cfg.aug_prob
        self.epochs = cfg.epochs
        self.class_weight = torch.Tensor(cfg.class_weight).to(self.device)

        # Models
        print("initialize model...")
        self.model_type = cfg.model_type
        self.optimizer_type = cfg.optimizer_type
        self.lr_sch_type = cfg.lr_sch_type
        self.dataset = cfg.dataset
        self.loss_eq = []
        self.loss_eq.append(DiceLoss(weight=self.class_weight))
        
        self.init_model = initialize_model(self.cfg)
        self.model = self.init_model.get_model(self.model_type)
        self.optimizer, self.lr_sch = self.init_model.get_optimizer(self.optimizer_type, self.lr_sch_type)
        self.grad_scaler = torch.amp.GradScaler(enabled=True)
        
        if self.cfg.mode == 'train':
            self.params = 0
            for p in self.model.parameters():
                self.params += p.numel()
            self.cfg.parameters = self.params
            
            print(self.model_type)
            self.report.write('\n' + str(self.model_type))
            print("The number of parameters: {}".format(self.params))
            self.report.write("\n The number of parameters: {}".format(self.params))
            self.report.write('\n' + str(self.model))
            
        if self.wandb:
        # wandb init
            self.project_name = self.dataset +'_'+ self.cfg.modality if self.dataset == 'Axial' else self.dataset
            self.experiment = wandb.init(entity='.', project=self.project_name, group='.',
                                         resume='allow', config=self.cfg)
            self.cfg = wandb.config
            wandb.define_metric("loss", summary="min")
            wandb.define_metric("mean_IoU", summary="max")
            wandb.define_metric("mean_Dc", summary="max")
        
    def train(self):
        # ====================================== Training ===========================================#
        model_score = 0.
        t = time.time()
        elapsed = 0.  # Time of inference

        # Model Train
        if os.path.isfile(self.model_path):
            Train_results = open(self.result_path + self.report_name + '_Train_result.csv', 'a', encoding='utf-8', newline='')
            twr = csv.writer(Train_results)
            
            Valid_results = open(self.result_path + self.report_name + '_Valid_result.csv', 'a', encoding='utf-8', newline='')
            vwr = csv.writer(Valid_results)
        else:
            Train_results = open(self.result_path + self.report_name + '_Train_result.csv', 'a', encoding='utf-8', newline='')
            twr = csv.writer(Train_results)

            Valid_results = open(self.result_path + self.report_name + '_Valid_result.csv', 'a', encoding='utf-8', newline='')
            vwr = csv.writer(Valid_results)
            twr.writerow(['Train_model', 'Model_type', 'Dataset', 'LR', 'Epochs', 'Aug_prob'])
            twr.writerow([self.report_name, self.model_type, self.dataset, self.lr, self.epochs, self.aug_prob])
            twr.writerow(['Epoch', 'LR', 'Loss', 'Acc', 'Rc', 'Pr', 'F1', 'IoU', 'Dc'])

            vwr.writerow(['Train_model', 'Model_type', 'Dataset', 'LR', 'Epochs', 'Aug_prob'])
            vwr.writerow([self.report_name, self.model_type, self.dataset, self.lr, self.epochs, self.aug_prob])
            vwr.writerow(['Epoch', 'LR', 'Loss', 'Acc', 'Rc', 'Pr', 'F1', 'IoU', 'Dc'])

        # Training
        self.Learner = Learner(self.cfg, self.model, self.optimizer, self.grad_scaler, self.loss_eq)
        results = [["Loss",[],[]], ["Acc",[],[]], ["Rc",[],[]], ["Pr",[],[]], ["F1",[],[]], ["IoU",[],[]], ["Dc",[],[]]]
        
        for epoch in range(self.epochs):
            # Print the report info
            print('\nEpoch [%d/%d], LR: [%0.5f] \n[Training]' % (epoch+1, self.epochs, self.lr))
            self.report.write('\nEpoch [%d/%d], LR: [%0.5f] \n[Training]' % (epoch+1, self.epochs, self.lr))
            
            evaluator = Evaluation(self.cfg.ch_out, classes=False)
            with tqdm(total=len(self.train_loader.dataset)) as pbar:
                for i, (image, GT, name) in enumerate(self.train_loader):
                    R = self.Learner.learning(image, GT, train='train')
                    
                    # Get metrices results
                    metrics = evaluator.metrics(R[0], R[1], R[2])
                    
                    pbar.update(image.shape[0])
                    pbar.set_postfix(**{'batch loss': R[2].item()})
                    
            metavg = evaluator.metrics_avg(metrics)
            for i in range(len(results)):
                results[i][1].append(metavg[i])
                
            print('\n[R] Loss: %.4f, Acc: %.4f, Rc: %.4f, Pr: %.4f, F1: %.4f, IoU: %.4f, Dc: %.4f' % (
                metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6]))
            self.report.write('\n[R] Loss: %.4f, Acc: %.4f, Rc: %.4f, Pr: %.4f, F1: %.4f, IoU: %.4f, Dc: %.4f' % (
                metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6]))
            twr.writerow(
                [epoch+1, self.lr, metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6]])
            
            # Clear unoccupied GPU memory after each epoch
            torch.cuda.empty_cache()
            
            # ========================== Validation ====================================#
            print('\n[Validating]')
            self.report.write('\n[Validating]')
            
            evaluator = Evaluation(self.cfg.ch_out, classes=True)
            with tqdm(total=len(self.valid_loader.dataset)) as pbar:
                for i, (image, GT, name) in enumerate(self.valid_loader):
                    R = self.Learner.learning(image, GT, train='valid')
                    
                    # Get metrices results
                    metrics = evaluator.metrics(R[0], R[1], R[2])
                    
                    pbar.update(image.shape[0])
                    pbar.set_postfix(**{'batch loss': R[2].item()})
                    
            metavg = evaluator.metrics_avg(metrics)
            for i in range(len(results)):
                if type(metavg[i]) == torch.Tensor:
                    results[i][2].append(metavg[i][-1])
                elif type(metavg[i]) == float:
                    results[i][2].append(metavg[i])
                    
            print('\n[R] Loss: %.4f, Acc: %.4f, Rc: %.4f, Pr: %.4f, F1: %.4f, IoU: %.4f, Dc: %.4f' % (
                metavg[0], metavg[1][-1], metavg[2][-1], metavg[3][-1], metavg[4][-1], metavg[5][-1], metavg[6][-1]))
            self.report.write('\n[R] Loss: %.4f, Acc: %.4f, Rc: %.4f, Pr: %.4f, F1: %.4f, IoU: %.4f, Dc: %.4f' % (
                metavg[0], metavg[1][-1], metavg[2][-1], metavg[3][-1], metavg[4][-1], metavg[5][-1], metavg[6][-1]))
            vwr.writerow([epoch+1, self.lr, metavg[0], metavg[1][-1], metavg[2][-1], metavg[3][-1], metavg[4][-1],
                          metavg[5][-1], metavg[6][-1]])
            
            # Decay learning rate
            self.lr_sch.step(metavg[4][-1])
            self.lr = self.optimizer.param_groups[0]['lr']
            
            # Save Best Model
            if metavg[4][-1] > model_score:
                model_score = metavg[4][-1]
                print('\nBest %s model score : %.4f' % (self.model_type, model_score))
                self.report.write('\nBest %s model score : %.4f' % (self.model_type, model_score))
                state_dict = self.model.state_dict()
                torch.save(state_dict, self.model_path)
                if self.wandb:
                    wandbBestMertics(self.dataset, metavg[5], metavg[6])
                    wandbTrainedModel(self.model_path, self.cfg)
                
            # Clear unoccupied GPU memory after each epoch
            torch.cuda.empty_cache()
            
            if self.wandb:
                wandbMetrics(self.dataset, self.experiment, self.lr, (epoch+1), metavg[0], metavg[5], metavg[6])
        displayfigures(results, self.result_path, self.report_name)
        
        if self.wandb:
            wandb.finish()
        Train_results.close()
        Valid_results.close()
        elapsed = time.time() - t
        print("\nElapsed time: %f seconds.\n\n" % elapsed)
        self.report.write("\nElapsed time: %f seconds.\n\n" % elapsed)
        self.report.close()

    def test(self):
        # ===================================== Test ====================================#
        
        # Load Trained Model
        if os.path.isfile(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu', weights_only=False))
            self.model = self.model.to(self.device)
            print('%s is Successfully Loaded from %s' % (self.model_type, self.model_path))
            self.report.write('\n%s is Successfully Loaded from %s' % (self.model_type, self.model_path))
        else:
            print("Trained model NOT found, Please train a model first")
            self.report.write("\nTrained model NOT found, Please train a model first")
            return

        self.Learner = Learner(self.cfg, self.model, self.optimizer, self.grad_scaler, self.loss_eq)
        results = [["Loss",[]], ["Acc",[]], ["Rc",[]], ["Pr",[]], ["F1",[]], ["IoU",[]], ["Dc",[]]]
        
        # Print the report info
        print('\n[Testing]')
        self.report.write('\n[Testing]')
        
        elapsed = 0.  # Time of inference
        evaluator = Evaluation(self.cfg.ch_out, classes=True)
        with tqdm(total=len(self.test_loader.dataset)) as pbar:
            for i, (image, GT, name) in enumerate(self.test_loader):
                t = time.time() # Time of inference
                R = self.Learner.learning(image, GT, train='test')
                elapsed = (time.time() - t)
                
                # Get metrices results
                metrics = evaluator.metrics(R[0], R[1], R[2])
                
                pbar.update(image.shape[0])
                pbar.set_postfix(**{'batch loss': R[2].item()})
                
        metavg = evaluator.metrics_avg(metrics)
        for i in range(len(results)):
            if type(metavg[i]) == torch.Tensor:
                results[i][1].append(metavg[i].tolist())
            elif type(metavg[i]) == float:
                results[i][1].append(metavg[i])
                
        elapsed = elapsed / (R[0].size(0))
        
        f = open(os.path.join(self.result_path, 'Test_result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow(['Report_file', 'Model_type', 'Dataset', 'Loss', 'Acc', 'Rc', 'Pr', 'F1', 'IoU', 'Dc',
             'Time of inference', 'LR', 'Epochs', 'Aug_prob'])
        wr.writerow([self.report_name, self.model_type, self.dataset, metavg[0], metavg[1], metavg[2], metavg[3], metavg[4],
             metavg[5], metavg[6], elapsed, self.lr, self.epochs, self.aug_prob])
        f.close()
        
        print('Results have been Saved')
        self.report.write('\nResults have been Saved\n\n')
        self.report.close()
