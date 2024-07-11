import torch
import numpy as np
import math
from copy import deepcopy
import torch.nn.functional as F
from torch import nn


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class HARDAloss():
    def __init__(self, vnums=4, loss_type='MSE'):
        self.vnums = vnums
        if loss_type == 'L1':
            self.loss = nn.L1Loss()
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss()
        else:
            pass
    
    def loss_rec(self, preds, x):
        x_mean = torch.mean(x, dim=1).unsqueeze(1)
        rec_img = self.rec(preds, x)
        loss = self.loss(rec_img, x_mean)
        return loss
    
    def rec_e(self, preds):
        e = preds[:,0,:,:].unsqueeze(1)
        return e
    
    def rec_T(self, preds):
        T = preds[:,1,:,:].unsqueeze(1)
        return T
    
    def rec_env(self, preds, x_mean):
        b, _, h, w = preds.shape
        V = preds[:,2:2+self.vnums,:,:]
        h_split_nums = int(math.sqrt(self.vnums))
        w_split_nums = self.vnums // h_split_nums
        assert h_split_nums * w_split_nums == self.vnums
        x_beta = F.avg_pool2d(x_mean, (h // h_split_nums, w // w_split_nums)).reshape(b, 1, self.vnums)
        v_pred = V.reshape(b, self.vnums, h*w)
        env = torch.matmul(x_beta, v_pred)
        env = env.view(b, 1, h, w)
        return env
    
    def rec(self, preds, x):
        x_mean = torch.mean(x, dim=1)
        e = self.rec_e(preds)
        T = self.rec_T(preds)
        env = self.rec_env(preds, x_mean)
        rec_img = torch.mul(e, T) + torch.mul(1-e, env)
        return rec_img
