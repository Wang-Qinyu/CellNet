# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:57:15 2022

@author: wangqinyu
"""
import os
import sys
from tabnanny import verbose
import torch
import numpy as np
from Loss import CtdetLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from dataset import ctDataset,ctDataset_DSB
import pickle

sys.path.append(r'./backbone')
from charnet import CharNet
from resnet_fpn import ResNet

os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
use_gpu = torch.cuda.is_available()

model = ResNet(50)
model = torch.nn.DataParallel(model)
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

loss_weight={'hm_weight':1,'wh_weight':0.1,'reg_weight':0.1,'mask_weight':0.5}
# loss_weight={'hm_weight':0.,'wh_weight':0.,'reg_weight':0.,'mask_weight':1}
criterion = CtdetLoss(loss_weight)

device = torch.device("cuda")
if use_gpu:
    model.cuda()

# model.load_state_dict(torch.load('./weights/OWN/resnet101fpn_1108_epoch_3.pth'))


model.train()

learning_rate = 1.25e-4
num_epochs = 60


optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=1.25e-4) 
lr_scheduler = StepLR(optimizer,
                      step_size=20,
                      gamma=0.5,
                      verbose=True)

train_dataset = ctDataset_DCCB(split='train')
train_loader = DataLoader(train_dataset,batch_size=8,shuffle=False,num_workers=4)  # num_workers是加载数据（batch）的线程数目

test_dataset = ctDataset_DCCB(split='val')
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4)
print('the dataset has %d images' % (len(train_dataset)))


num_iter = 0

best_test_loss = np.inf 
#! write logs
writer = SummaryWriter('./visualization/DCCB')
for epoch in range(0,num_epochs):
    model.train()
    
    # total_loss = 0.
    for i, sample in enumerate(train_loader):
        # print(sample.keys())
        for k in sample:
            sample[k] = sample[k].to(device=device, non_blocking=True)
        # print(sample[k].shape)
        pred = model(sample['input'])
        
        # loss = criterion(pred, sample)
        hm_loss, wh_loss, reg_loss, mask_loss = criterion(pred, sample) 
        loss = hm_loss + wh_loss + reg_loss + mask_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] total_loss: %.4f hm_oss: %.4f  wh_loss: %.4f reg_loss: %.4f mask_loss: %.4f' 
            %(epoch, num_epochs, i+1, len(train_loader), 
              loss.item(), 
              hm_loss.item(), 
              wh_loss.item(), 
              reg_loss.item(), 
              mask_loss))
            num_iter += 1
            
        # if (i+1) % 10000 == 0:
        #     torch.save(model.state_dict(),'./weights/OWN/resnet50fpn_0107_epoch_{}_{}.pth'.format(epoch,i+1),_use_new_zipfile_serialization=False)
        # writer.add_scalar("train/dsb_shrink06_total", loss.data, i + (epoch-1) * len(train_loader))
        # writer.add_scalar("train/dsb_shrink06_hm", hm_loss.data, i + (epoch-1) * len(train_loader))
        # writer.add_scalar("train/dsb_shrink06_wh", wh_loss.data, i + (epoch-1) * len(train_loader))
        # writer.add_scalar("train/dsb_shrink06_reg", reg_loss.data, i + (epoch-1) * len(train_loader))
        # writer.add_scalar("train/dsb_shrink06_mask", mask_loss.data, i + (epoch-1) * len(train_loader)) 
    
    
    validation_loss = 0.0
    model.eval()
    for i, sample in enumerate(test_loader):
        if use_gpu:
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)

        pred = model(sample['input'])
        # loss = criterion(pred, sample)
        hm_loss, wh_loss, reg_loss, mask_loss = criterion(pred, sample) 
        val_loss = hm_loss + wh_loss + reg_loss + mask_loss

        validation_loss += val_loss.item()
         
    validation_loss /= len(test_loader)
    writer.add_scalar("test_loss/DSB_test0607_1", validation_loss, epoch+1)
    
    # writer.add_scalar("test/total", loss.data, epoch)
    # writer.add_scalar("test/hm", hm_loss.data, i + (epoch-1) * len(train_loader))
    # writer.add_scalar("test/wh", wh_loss.data, i + (epoch-1) * len(train_loader))
    # writer.add_scalar("test/reg", reg_loss.data, i + (epoch-1) * len(train_loader))
    # writer.add_scalar("test/mask", mask_loss.data, i + (epoch-1) * len(train_loader))
    
   
    torch.save(model.state_dict(),'./weights/epoch_{}.pth'.format(epoch),_use_new_zipfile_serialization=False)
    lr_scheduler.step()
