# -*- coding: utf-8 -*-
"""
Created on May 8 2025

@author: Omar Al-maqtari
"""
import torch

# R : Model Result
# GT : Ground Truth

class Evaluation(object):
    def __init__(self, n_classes, classes=False):
        self.n_classes = n_classes
        self.classes = classes
        self.loss = 0.
        self.Acc = 0.	# Accuracy
        self.Rc = 0.	# Recall (Sensitivity)
        self.Pr = 0. 	# Precision
        self.F1 = 0.    # F1-score
        self.IoU = 0.   # Intersection over Union (Jaccard Index)
        self.Dc = 0.    # Dice coefficient
        self.length = 0
        
    @torch.no_grad()
    def get_results(self, R, GT):
        Acc = []
        Rc = []
        Pr = []
        F1 = []
        IoU = []
        Dc = []

        R_copy = torch.argmax(R, axis=1).reshape(-1).float()
        GT_copy = GT.squeeze(1).view(-1).float()
        for cls in range(self.n_classes):
            tp = torch.sum((R_copy==cls)&(GT_copy==cls))
            tn = torch.sum((R_copy!=cls)&(GT_copy!=cls))
            fp = torch.sum((R_copy==cls)&(GT_copy!=cls))
            fn = torch.sum((R_copy!=cls)&(GT_copy==cls))
            
            Acc.append(((tp + tn) / (tp + fp + fn + tn)))
            Rc.append((tp / (tp + fn + 1e-12)))
            Pr.append((tp / (tp + fp + 1e-12)))
            F1.append(((2. * Rc[-1] * Pr[-1]) / (Rc[-1] + Pr[-1] + 1e-12)))
            IoU.append((tp / (tp + fp + fn + 1e-12)))
            Dc.append(((2. * tp) / (tp + tp + fp + fn + 1e-12)))
            
        Acc_l = torch.Tensor(Acc)
        Rc_l = torch.Tensor(Rc)
        Pr_l = torch.Tensor(Pr)
        F1_l = torch.Tensor(F1)
        IoU_l = torch.Tensor(IoU)
        Dc_l = torch.Tensor(Dc)
        if self.classes:
            Acc = torch.cat([Acc_l, Acc_l.mean().unsqueeze(0)])
            Rc = torch.cat([Rc_l, Rc_l.mean().unsqueeze(0)])
            Pr = torch.cat([Pr_l, Pr_l.mean().unsqueeze(0)])
            F1 = torch.cat([F1_l, F1_l.mean().unsqueeze(0)])
            IoU = torch.cat([IoU_l, IoU_l.mean().unsqueeze(0)])
            Dc = torch.cat([Dc_l, Dc_l.mean().unsqueeze(0)])
        else:
            Acc = Acc_l.mean()
            Rc = Rc_l.mean()
            Pr = Pr_l.mean()
            F1 = F1_l.mean()
            IoU = IoU_l.mean()
            Dc = Dc_l.mean()
        
        return [Acc, Rc, Pr, F1, IoU, Dc]
    
    def metrics(self, R, GT, total_loss):
        R = R.detach()
        GT = GT.detach()
        results = self.get_results(R, GT)
        self.loss += total_loss.detach().item()
        self.Acc += results[0]
        self.Rc += results[1]
        self.Pr += results[2]
        self.F1 += results[3]
        self.IoU += results[4]
        self.Dc += results[5]
        self.length += 1
        
        return [self.loss, self.Acc, self.Rc, self.Pr, self.F1, self.IoU, self.Dc, self.length]
            
    def metrics_avg(self, metric):
        loss = (metric[0]/metric[-1])
        Acc = (metric[1]/metric[-1])*100
        Rc = (metric[2]/metric[-1])*100
        Pr = (metric[3]/metric[-1])*100
        F1 = (metric[4]/metric[-1])*100
        IoU = (metric[5]/metric[-1])*100
        Dc = (metric[6]/metric[-1])*100
        
        return [loss, Acc, Rc, Pr, F1, IoU, Dc]
            
