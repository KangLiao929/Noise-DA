import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    
class TripletMarginLoss(nn.Module):
    def __init__(self, margin=0.1, p=2.0, eps=1e-6, swap=False, reduction='mean'):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction

    def forward(self, anchor, positive, negative):
        pos_dist = F.mse_loss(anchor, positive)
        neg_dist = F.mse_loss(anchor, negative)
        
        if self.swap:
            an_dist = F.mse_loss(positive, negative)
            neg_dist = torch.min(neg_dist, an_dist)

        loss = torch.relu(pos_dist - neg_dist + self.margin)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def mse_loss(output, target):
    return F.mse_loss(output, target)

def mse_loss_alpha(output, alpha):
    return alpha*output

def charbonnier_loss(output, target):
    charbonnierloss = CharbonnierLoss()
    return charbonnierloss(output, target)

def final_loss(loss1, loss2):
    return loss1 + loss2

def cal_triplet_margin_loss(anchor, positive, negative):
    loss = TripletMarginLoss()
    return loss(anchor, positive, negative)