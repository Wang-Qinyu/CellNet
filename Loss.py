# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:59:17 2020

@author: wangqinyu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * \
        neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    #! Support by https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, pred_tensor, target_tensor):
        return self.neg_loss(pred_tensor, target_tensor)


class MaskLoss(nn.Module):
    ''' for compute hard mask loss'''
    def __init__(self) -> None:
        super(MaskLoss,self).__init__()
        
    def forward(self,pred_tensor,target_tensor):
        # print('=================',pred_tensor.shape,target_tensor.shape)
        # pred_tensor.sigmoid_()
        a = 2 * torch.sum(pred_tensor * target_tensor)
        b = torch.sum(pred_tensor.pow(2)) + torch.sum(target_tensor.pow(2))
        loss = 1- a/b
        # print('======================',a,b)
        return loss
        # return (torch.tensor(0.0 ,requires_grad=True)) if loss == 0 else loss
        # return self.hardMask(pred_tensor,target_tensor)

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, mask, ind, target):
        pred = _transpose_and_gather_feat(pred, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)  # 每个目标的平均损失
        return loss


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

# def _relu(x):
#     y = torch.clamp(x.relu_(), min = 1e-4, max=math.pi-0.01)
#     return y


def _relu(x):
    y = torch.clamp(x, min=-math.pi, max=math.pi)
    return y


class CtdetLoss(torch.nn.Module):
    # loss_weight={'hm_weight':1,'wh_weight':0.1,'reg_weight':0.1}
    def __init__(self, loss_weight):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.loss_weight = loss_weight
        # self.maskLoss = BinaryDiceLoss()  #torch.nn.MSELoss()
        self.maskLoss = MaskLoss()  #torch.nn.MSELoss()

    def forward(self, pred_tensor, target_tensor):
        hm_weight = self.loss_weight['hm_weight']
        wh_weight = self.loss_weight['wh_weight']
        reg_weight = self.loss_weight['reg_weight']
        mask_weight = self.loss_weight['mask_weight']
        # print('----=====',pred_tensor.keys())
        # print('############',target_tensor['mask'].unsqueeze(1).shape,pred_tensor['mask'].shape)
        # print('############',target_tensor['hm'].shape,pred_tensor['hm'].shape)
        # print('############',type(target_tensor['mask']),type(pred_tensor['mask']))
        # ang_weight = self.loss_weight['ang_weight']
#        print(pred_tensor['hm'].size())
        hm_loss, wh_loss, off_loss = 0., 0., 0.
        mask_loss = 0.
        # print(_sigmoid(pred_tensor['mask']).shape,target_tensor['mask'].shape)
        mask_loss += self.maskLoss(_sigmoid(pred_tensor['mask']),target_tensor['mask'])
        # print(pred_tensor['mask'].dtype,target_tensor['mask'].dtype)
        
        pred_tensor['hm'] = _sigmoid(pred_tensor['hm'])
#        print(target_tensor['hm'].size())
        hm_loss += self.crit(pred_tensor['hm'], target_tensor['hm'])
        # if ang_weight > 0:
        #     pred_tensor['ang'] = _relu(pred_tensor['ang'])
        #     ang_loss += self.crit_wh(pred_tensor['ang'], target_tensor['reg_mask'],
        #                              target_tensor['ind'], target_tensor['ang'])

        if wh_weight > 0:
            wh_loss += self.crit_wh(pred_tensor['wh'], target_tensor['reg_mask'],
                                    target_tensor['ind'], target_tensor['wh'])
        if reg_weight > 0:
            off_loss += self.crit_reg(pred_tensor['reg'], target_tensor['reg_mask'],
                                      target_tensor['ind'], target_tensor['reg'])
        # print(hm_loss,wh_loss,off_loss,mask_loss)
        # return hm_weight * hm_loss + wh_weight * wh_loss + reg_weight * off_loss  + mask_weight * mask_loss
        return hm_weight * hm_loss, wh_weight * wh_loss, reg_weight * off_loss, mask_weight * mask_loss

