# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 18:11:15 2020

@author: wangqinyu
"""

import os
import cv2
import math
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import pycocotools.coco as coco

from torchvision.utils import save_image
import shutil
import torch
from PIL import Image

class ctDataset(data.Dataset):
    num_classes = 1
    default_resolution = [512,512] #! This is a alert

    # ! PSM
    mean = np.array([0.72720685,0.73073854,0.63989986], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.0348947,0.04314667,0.03857797], dtype=np.float32).reshape(1, 1, 3)
    
    def __init__(self, data_dir='data', split='train'):
        self.data_dir = os.path.join(data_dir, 'PSM')
        self.img_dir = os.path.join(self.data_dir, 'images')
        
        self.split =split
        try:
            if split == 'train':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'train_add.json') #train_0903.json
            elif split == 'val':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'test_add.json')
        except:
            print('No any data!')

        self.max_objs = 128
        self.class_name = ['obj']
        self._valid_ids = [i for i in range(1,self.num_classes + 1)]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],dtype=np.float32)
        self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

        self.split = split
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        self.transforms = transforms.Compose([
                transforms.ColorJitter(1,1,1),
                transforms.GaussianBlur(1,(0.1,2)),
                transforms.RandomInvert(p=0.3),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.3)
                ])
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        
        # print(file_name)
        # shutil.copyfile(os.path.join(self.img_dir,file_name),'./buff/'+file_name)
        
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids) 
        
        num_objs = min(len(anns), self.max_objs)
         
        
        if self.split == 'train':
            img = Image.open(img_path)
            img = self.transforms(img)
            img = np.array(img)[:,:,::-1]
        else:
            img = cv2.imread(img_path) 
        
        
        height, width = img.shape[0], img.shape[1]  
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # ?????????
    
        keep_res = False #
        if keep_res:
            input_h = (height | 31) + 1   
            input_w = (width | 31) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0  
            input_h, input_w = 512, 512 

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,(input_w, input_h),flags=cv2.INTER_LINEAR) 
        inp = (inp.astype(np.float32) / 255.)  

        #?????????
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1) 
        
        down_ratio = 4 
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = self.num_classes 
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])  

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)   
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        # ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32) 
        ind = np.zeros((self.max_objs), dtype=np.int64) 
        
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8) 
        draw_gaussian = draw_umich_gaussian
        
        mask = np.zeros((1,128,128),dtype=np.float32)
        
        for k in range(num_objs):  # num_objs?????????????????????  
            ann = anns[k]  # ???????????????????????????
               
            bbox = coco_box_to_bbox(ann['bbox']) 
            
            #! mask label
            c_x, c_y = int(ann['bbox'][0]/4), int(ann['bbox'][1]/4)
            mask_w, mask_h = int(ann['bbox'][2]/4), int(ann['bbox'][3]/4)
            shrink = 0.4
            mask_w = int(shrink * mask_w)
            mask_h = int(shrink * mask_h)
            mask[:,c_y-mask_h//2:c_y+mask_h//2,c_x-mask_w//2:c_x+mask_w//2] = 1.
            
            cls_id = int(self.cat_ids[ann['category_id']]) 
            bbox[:2] = affine_transform(bbox[:2], trans_output)    # ???box??????????????? 128*128????????????
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)  
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            # ????????????????????????????????????resize???????????????????????????
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))  
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) 
                ct_int = ct.astype(np.int32) 
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                # ang[k] = 1. * an
                ind[k] = ct_int[1] * output_w + ct_int[0]  
                reg[k] = ct - ct_int
                reg_mask[k] = 1
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,'mask':mask}
        reg_offset_flag = True #
        if reg_offset_flag:
            ret.update({'reg': reg})
        # print(ret['reg_mask'].shape,type(ret['reg_mask']))
        # mask = torch.from_numpy(mask)
        # save_image(mask,'./buff/{}.png'.format(file_name.split('.')[0]))
        return ret


class ctDataset_DSB(data.Dataset):
    num_classes = 1
    default_resolution = [512,512] #! This is a alert

    # ! PSM
    mean = np.array([0.0523233,0.0523233,0.0523233], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.05780539,0.05780539,0.05780539], dtype=np.float32).reshape(1, 1, 3)
    
    def __init__(self, data_dir='data', split='train'):
        self.data_dir = os.path.join(data_dir, 'DSB')
        self.img_dir = os.path.join(self.data_dir, 'images')
        
        self.split =split
        try:
            if split == 'train':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'train_DSB.json') #train_0903.json
            elif split == 'val':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'test_DSB.json')
        except:
            print('No any data!')

        self.max_objs = 128
        self.class_name = ['obj']
        self._valid_ids = [i for i in range(1,self.num_classes + 1)]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],dtype=np.float32)
        self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

        self.split = split
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        self.transforms = transforms.Compose([
                transforms.ColorJitter(1,1,1),
                transforms.GaussianBlur(1,(0.1,2)),
                # transforms.RandomInvert(p=0.3),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.3)
                ])
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        
        # print(file_name)
        # shutil.copyfile(os.path.join(self.img_dir,file_name),'./buff/'+file_name)
        
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids) 
        
        num_objs = min(len(anns), self.max_objs)
         
        
        # if self.split == 'train':
        #     img = Image.open(img_path).convert('RGB')
        #     img = self.transforms(img)
        #     img = np.array(img)[:,:,::-1]
        # else:
        #     img = cv2.imread(img_path)[:,:,:3]
        img = cv2.imread(img_path)[:,:,:3]
        
        
        height, width = img.shape[0], img.shape[1]  
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # ?????????
    
        keep_res = False #
        if keep_res:
            input_h = (height | 31) + 1   
            input_w = (width | 31) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0  
            input_h, input_w = 512, 512 

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,(input_w, input_h),flags=cv2.INTER_LINEAR) 
        inp = (inp.astype(np.float32) / 255.)  

        #?????????
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1) 
        
        down_ratio = 4 
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = self.num_classes 
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])  

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)   
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        # ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32) 
        ind = np.zeros((self.max_objs), dtype=np.int64) 
        
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8) 
        draw_gaussian = draw_umich_gaussian
        
        # mask = np.zeros((1,128,128),dtype=np.float32)
        
        for k in range(num_objs):  # num_objs?????????????????????  
            ann = anns[k]  # ???????????????????????????
               
            bbox = coco_box_to_bbox(ann['bbox']) 
            
            #! mask label
            c_x, c_y = int(ann['bbox'][0]/4), int(ann['bbox'][1]/4)
            mask_w, mask_h = int(ann['bbox'][2]/4), int(ann['bbox'][3]/4)
            shrink = 0.4
            mask_w = int(shrink * mask_w)
            mask_h = int(shrink * mask_h)
            # mask[:,c_y-mask_h//2:c_y+mask_h//2,c_x-mask_w//2:c_x+mask_w//2] = 1.
            
            cls_id = int(self.cat_ids[ann['category_id']]) 
            bbox[:2] = affine_transform(bbox[:2], trans_output)    # ???box??????????????? 128*128????????????
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)  
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            # ????????????????????????????????????resize???????????????????????????
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))  
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) 
                ct_int = ct.astype(np.int32) 
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                # ang[k] = 1. * an
                ind[k] = ct_int[1] * output_w + ct_int[0]  
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                
        #! load mask
        mask_path = os.path.join(self.img_dir, '_' + file_name.split('.')[0] + '.png')
        # print(mask_path)
        img_mask = Image.open(mask_path).convert('L').resize((128,128))
        # print(mask_path.size)
        mask = np.array(img_mask)/255.
        mask = np.expand_dims(mask,axis=0)
        # print(mask.shape,mask.max())
        
        # mask = (mask > 0.7) * 1.
        # print('========= DSB ======',mask.shape)
        
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,'mask':mask}
        reg_offset_flag = True #
        if reg_offset_flag:
            ret.update({'reg': reg})
        # print(ret['reg_mask'].shape,type(ret['reg_mask']))
        # mask = torch.from_numpy(mask)
        # save_image(mask,'./buff/{}.png'.format(file_name.split('.')[0]))
        return ret


class ctDataset_DCCB(data.Dataset):
    num_classes = 1
    default_resolution = [512,512] #! This is a alert

    # ! PSM
    mean = np.array([0.71520138,0.69587003,0.73721376], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.07304651,0.10050465,0.0589885], dtype=np.float32).reshape(1, 1, 3)
    
    def __init__(self, data_dir='data', split='train'):
        self.data_dir = os.path.join(data_dir, 'DCCB')
        
        
        self.split =split
        try:
            if split == 'train':
                self.img_dir = os.path.join(self.data_dir, 'images','train')
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'train_DCCB.json') #train_0903.json
            elif split == 'val':
                self.img_dir = os.path.join(self.data_dir, 'images','test')
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'test_DCCB.json')
        except:
            print('No any data!')

        self.max_objs = 128
        self.class_name = ['obj']
        self._valid_ids = [i for i in range(1,self.num_classes + 1)]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],dtype=np.float32)
        self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

        self.split = split
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)


        self.transforms = transforms.Compose([
                transforms.ColorJitter(0.3,0.3,0.5),
                transforms.GaussianBlur(1,(0.1,2)),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.3)
                ])
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        
        # print(file_name)
        # shutil.copyfile(os.path.join(self.img_dir,file_name),'./buff/'+file_name)
        
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids) 
        
        num_objs = min(len(anns), self.max_objs)
         
        
        if self.split == 'train':
            img = Image.open(img_path).convert('RGB')
            img = self.transforms(img)
            img = np.array(img)[:,:,::-1]
        else:
            img = cv2.imread(img_path)[:,:,:3]
        # img = cv2.imread(img_path)[:,:,:3]
        
        height, width = img.shape[0], img.shape[1]  
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # ?????????
    
        keep_res = False #
        if keep_res:
            input_h = (height | 31) + 1   
            input_w = (width | 31) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0  
            input_h, input_w = 512, 512 

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,(input_w, input_h),flags=cv2.INTER_LINEAR) 
        inp = (inp.astype(np.float32) / 255.)  

        #?????????
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1) 
        
        down_ratio = 4 
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = self.num_classes 
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])  

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)   
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        # ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32) 
        ind = np.zeros((self.max_objs), dtype=np.int64) 
        
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8) 
        draw_gaussian = draw_umich_gaussian
        
        mask = np.zeros((1,128,128),dtype=np.float32)
        
        for k in range(num_objs):  # num_objs?????????????????????  
            ann = anns[k]  # ???????????????????????????

            #!  alter start
            bbox = coco_box_to_bbox(ann['bbox']) 
            # bbox = np.array([ann['bbox'][0],ann['bbox'][1],ann['bbox'][2],ann['bbox'][3]])
            #!  alter end
            
            #! mask label
            c_x, c_y = int(ann['bbox'][0]/4), int(ann['bbox'][1]/4)
            mask_w, mask_h = int(ann['bbox'][2]/4), int(ann['bbox'][3]/4)
            shrink = 0.5
            mask_w = int(shrink * mask_w)
            mask_h = int(shrink * mask_h)
            mask[:,c_y-mask_h//2:c_y+mask_h//2,c_x-mask_w//2:c_x+mask_w//2] = 1.
            
            cls_id = int(self.cat_ids[ann['category_id']]) 
            bbox[:2] = affine_transform(bbox[:2], trans_output)    # ???box??????????????? 128*128????????????
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)  
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            # ????????????????????????????????????resize???????????????????????????
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))  
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) 
                ct_int = ct.astype(np.int32) 
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                # ang[k] = 1. * an
                ind[k] = ct_int[1] * output_w + ct_int[0]  
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                
        #! load mask
        # mask_path = os.path.join(self.img_dir, '_' + file_name.split('.')[0] + '.png')
        # print(mask_path)
        # img_mask = Image.open(mask_path).convert('L').resize((128,128))
        # print(mask_path.size)
        # mask = np.array(img_mask)/255.
        # mask = np.expand_dims(mask,axis=0)
        # print(mask.shape,mask.max())
        
        # mask = (mask > 0.7) * 1.
        # print('========= DSB ======',mask.shape)
        
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,'mask':mask}
        reg_offset_flag = True #
        if reg_offset_flag:
            ret.update({'reg': reg})
        # print(ret['reg_mask'].shape,type(ret['reg_mask']))
        # mask = torch.from_numpy(mask)
        # save_image(mask,'./buff/{}.png'.format(file_name.split('.')[0]))
        return ret



def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)
    
def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2
    
def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])
    
def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha
    
def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)
# ????????????
def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)
    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)
    
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result    
    
def get_affine_transform(center,scale,rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32),inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1) 
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right] 
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right] 
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
    
def coco_box_to_bbox(box):
    bbox = np.array([box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2],dtype=np.float32)
    # ang = float(box[4])
    return bbox

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter*2+1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)
  
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
