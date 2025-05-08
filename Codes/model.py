# -*- coding: utf-8 -*-
"""
Created on May 8 2025

@author: Omar Al-maqtari
"""
import torch
from torch import nn
import torch.nn.functional as F

from mamba import MambaBlock
from EEAtten import EEA
from frcm import FRCM


# Squeeze and Excitation Attention
class SEM(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, reduction, kernel_size=1, bias=False),
            nn.PReLU(num_parameters=reduction,init=0.02),
            nn.Conv2d(reduction, ch_in, kernel_size=1, bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        
        return x * y.expand_as(x)
    
    
# Convolution layer
class iConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, dilation=1, groups=1, bias=False, act='identity'):
        super().__init__()
        self.iconv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel, padding="same", dilation=dilation, groups=groups, bias=bias),
            nn.GroupNorm(4,ch_out)
            )
        if act == 'identity':
            self.act = nn.Identity()
        elif act == 'prelu':
            self.act = nn.PReLU(num_parameters=ch_out,init=0.02)
        elif act == 'relu':
            self.act = nn.ReLU(True)
            
    def forward(self, x):
        return self.act(self.iconv(x))
    
    
# Main Block
class Block(nn.Module):
    def __init__(self, ch_in, ch_out, sa=0, ma=0):
        super(Block, self).__init__()
        self.sa = sa
        self.ma = ma
        ch_mid = ch_out//4
        self.softmax = nn.Softmax(-1)
        
        self.squeeze = iConv(ch_in, ch_mid, 1, 1, 4, act='prelu')
        
        # Maxpooling
        self.q = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            iConv(ch_mid, ch_mid, 1, 1, 2)
            )
        
        # dilated 3x3 conv branch 
        self.k = iConv(ch_mid, ch_mid, 3, 2, 2)
        
        # 3x3 conv branch
        self.c = iConv(ch_mid, ch_mid*3, 3, 1, 2)
        
        if self.sa > 0:
            self.prelu_f = nn.PReLU(num_parameters=ch_mid, init=0.01)
            self.prelu_b = nn.PReLU(num_parameters=ch_mid, init=0.9)
            self.EEA = EEA(ch_mid).to('cuda:0')
            self.sem1 = SEM(ch_mid, reduction=4)
            if self.ma > 0:
                self.Mamba = MambaBlock(ch_mid, self.ma).to('cuda:0')
                
            self.nt = nn.Sequential(nn.GroupNorm(4, ch_mid),
                                    nn.PReLU(num_parameters=ch_mid, init=0.02)
                                    )
        self.shortcut = iConv(ch_in, ch_out, 1, 1, 8, act='prelu') if ch_in != ch_out else nn.Identity() 
        self.sem2 = SEM(ch_out, reduction=4)
        self.sem3 = SEM(ch_out, reduction=4)
        
    # non-linear V vector projection
    def v(self, q, k):
        return -self.prelu_b(-self.prelu_f(q + k))
    
    # Channel Self-attention
    def CSA(self, q, k):
        b, c, h, w = q.size()
        
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = self.v(q, k)
        
        _, c, _ = q.shape
        q = torch.matmul(q, k.transpose(-1, -2)) * (c ** -0.5)
        q = self.softmax(q)
        v = torch.matmul(q, v).contiguous().view(b, c, h, w)
        
        return v
    
    # Patch Self-attention
    def PSA(self, q, k):
        b, c, h, w = q.shape
        p = (h//self.sa)**2
        q = q.view(b, c, p, self.sa**2)
        k = k.view(b, c, p, self.sa**2)
        v = self.v(q, k)
        
        _, _, p, _ = q.shape
        q = torch.matmul(q, k.transpose(-1, -2)) * (p ** -0.5)
        q = self.softmax(q)
        v = torch.matmul(q, v).contiguous().view(b, c, h, w)
        
        return v
    
    # Multiscale Self-attention Module
    def MSM(self, x):
        q = self.q(x)
        k = self.k(x)
        if self.sa > 0:
            v = self.CSA(q, k) + self.PSA(q, k) + self.EEA(x) + self.sem1(x)
            if self.ma > 0:
                v = v + self.Mamba(x)
            return (q + k) + self.nt(v)
        else:
            v = (q + k)
            return v
        
    def forward(self, x):
        x_s = self.squeeze(x)
        
        v = self.MSM(x_s)
        x1 = self.c(v)
        
        x1 = torch.cat([x_s+v, x1], 1)
        x1 = self.sem2(x1) + x1
        
        x = self.shortcut(x)
        x = self.sem3(x) + x
        
        return x + x1
    
    
class Encoder(nn.Module):
    def __init__(self, img_in, ch_in):
        super(Encoder, self).__init__()
        self.E1 = nn.ModuleList([])
        self.E1.append(nn.ModuleList([
            nn.Conv2d(img_in, ch_in[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, ch_in[0]),
            Block(ch_in[0], ch_in[0]),
            Block(ch_in[0], ch_in[0])
            ]))
        self.E2 = nn.ModuleList([])
        self.E2.append(nn.ModuleList([
            Block(ch_in[0], ch_in[1], 8, 8),
            Block(ch_in[1], ch_in[1], 8),
            Block(ch_in[1], ch_in[1])
            ]))
        self.E3 = nn.ModuleList([])   
        self.E3.append(nn.ModuleList([
            Block(ch_in[1], ch_in[2], 4, 4),
            Block(ch_in[2], ch_in[2], 4),
            Block(ch_in[2], ch_in[2], 4),
            Block(ch_in[2], ch_in[2])
            ]))
        self.E4 = nn.ModuleList([])
        self.E4.append(nn.ModuleList([
            Block(ch_in[2], ch_in[3], 2, 2),
            Block(ch_in[3], ch_in[3], 2),
            Block(ch_in[3], ch_in[3])
            ]))
        
        self.P = nn.MaxPool2d(2,2)
        self.d1 = nn.Dropout(0.3)
        self.d2 = nn.Dropout(0.2)
        self.d3 = nn.Dropout(0.1)
        
    def forward(self, x):
        x1 = []
        for (conv, gn, e1b1, e1b2) in self.E1:
            x1.append(e1b2(e1b1(gn(conv(x)))))
        x2 = []
        for (e2b1, e2b2, e2b3) in self.E2:
            x2.append(self.d1(self.P(x1[-1])))
            x2.append(e2b3(e2b2(e2b1(x2[-1]))))
        x3 = []
        for (e3b1, e3b2, e3b3, e3b4) in self.E3:
            x3.append(self.d2(self.P(x2[-1])))
            x3.append(e3b4(e3b3(e3b2(e3b1(x3[-1])))))
        x4 = []
        for (e4b1, e4b2, e4b3) in self.E4:
            x4.append(self.d3(self.P(x3[-1])))
            x4.append(e4b3(e4b2(e4b1(x4[-1]))))
        
        return x1, x2, x3, x4
    
    
class Decoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Decoder, self).__init__()
        self.D1 = nn.ModuleList([])
        self.D1.append(nn.ModuleList([
            iConv(ch_in[0], ch_in[0]//2, 1, 1, 1, act='prelu'),
            iConv(ch_in[0], ch_in[0]//2, 1, 2, 4, act='prelu'),
            iConv(ch_in[0], ch_in[0], 1, 1, 2, act='prelu')
            ]))
        self.D2 = nn.ModuleList([])
        self.D2.append(nn.ModuleList([
            iConv(ch_in[1], ch_in[1]//2, 1, 1, 1, act='prelu'),
            iConv(ch_in[1], ch_in[1]//2, 1, 2, 4, act='prelu'),
            nn.MaxPool2d(3, stride=1, padding=1),
            iConv(ch_in[1], ch_in[0], 3, 1, 2, act='prelu')
            ]))
        self.D3 = nn.ModuleList([])
        self.D3.append(nn.ModuleList([
            iConv(ch_in[2], ch_in[2]//2, 1, 1, 1, act='prelu'),
            iConv(ch_in[2], ch_in[2]//2, 1, 2, 4, act='prelu'),
            nn.MaxPool2d(3, stride=1, padding=1),
            iConv(ch_in[2], ch_in[1], 1, 1, 4, act='prelu')
            ]))
        self.D4 = nn.ModuleList([iConv(ch_in[3], ch_in[2], 1, 1, 4, act='prelu')])
        
        self.FRCM = FRCM(ch_ins=[ch_in[0], ch_in[0], ch_in[1], ch_in[2], ch_in[3]], ch_out=4)
        
        ch_in = ch_in[0]+24
        self.sem = SEM(ch_in, reduction=4)
        self.conv = iConv(ch_in, 64, 3, 1, 64//8, True, act='relu')
        self.out = nn.Sequential(
                iConv(64, 64, 1, 1, 1, False, act='relu'),
                nn.Conv2d(64, ch_out, kernel_size=1, bias=False)
                )
        self.d = nn.Dropout(0.1)
        
    def forward(self, x, img_shape):
        x1, x2, x3, x4 = x
        
        for d4c in self.D4:
            x = d4c(x4[-1])
            x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
            
        for (d3c1a, d3c1b, mp, d3c2) in self.D3:
            x1_ = torch.cat([d3c1a(x3[-1]), d3c1b(x3[-1])], 1)
            x = d3c2(mp(x1_ + x))
            x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
            
        for (d2c1a, d2c1b, mp, d2c2) in self.D2:
            x1_ = torch.cat([d2c1a(x2[-1]), d2c1b(x2[-1])], 1)
            x = d2c2(mp(self.d(x1_) + x))
            x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
            
        for (d1c1a, d1c1b, d1c2) in self.D1:
            x1_ = torch.cat([d1c1a(x1[-1]), d1c1b(x1[-1])], 1)
            x = d1c2(self.d(x1_) + x)
            x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
            
        sides = self.FRCM([x, x1[-1], x2[-1], x3[-1], x4[-1]], img_shape)
        x = torch.cat([x,sides],1)
        
        return self.out(self.conv(x + self.sem(x)))
    

# Model
class M(nn.Module):
    def __init__(self, img_in, ch_in, segout):
        super(M, self).__init__()
        
        self.E = Encoder(img_in, ch_in)
        self.D = Decoder(ch_in, segout)
    
    def forward(self, x):
        img_shape = x.shape[2:]
        
        x = self.E(x)
        x = self.D(x, img_shape)
        
        return x


class Learner(object):
    def __init__(self, cfg, model, optimizer, grad_scaler, loss_eq):
        self.cfg = cfg
        self.model = model.to(self.cfg.cuda)
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.DiceLoss = loss_eq[0].to(self.cfg.cuda)
        
    def loss(self, R, GT):
        loss = self.DiceLoss(
            F.softmax(R, dim=1).float(),
            F.one_hot(GT.squeeze(1), self.cfg.ch_out).permute(0, 3, 1, 2).float())
        
        return loss
    
    def learning(self, image, GT, train='test'):
        if train == 'train':
            self.model.train(True)
            image = image.to(self.cfg.cuda)
            GT = GT.to(self.cfg.cuda)
            
            with torch.autocast(self.cfg.cuda, enabled=True):
                R = self.model(image)
                loss = self.loss(R, GT)
                
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                    
        else:
            self.model.train(False)
            with torch.no_grad():
                image = image.to(self.cfg.cuda)
                GT = GT.to(self.cfg.cuda)
                
                with torch.autocast(self.cfg.cuda, enabled=True):
                    R = self.model(image)
                    loss = self.loss(R, GT)
                    
        return [R, GT, loss]
    
    