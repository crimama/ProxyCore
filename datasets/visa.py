from glob import glob 
import os 

import numpy as np 
import pandas as pd 

import cv2 
from PIL import Image 

import torch 
from torch.utils.data import Dataset

class VISA(Dataset):
    '''
    Example 
        df = get_df(
            datadir       = datadir ,
            class_name    = class_name,
            anomaly_ratio = anomaly_ratio
        )
        trainset = MVTecAD(
            df           = df,
            train_mode   = 'train',
            transform    = train_augmentation,
            gt_transform = gt_augmentation,
            gt           = True 
        )
    '''
    def __init__(self, df: pd.DataFrame, train_mode:str, transform, gt_transform, gt=True, idx=False):
        '''
        train_mode = ['train','valid','test']
        '''
        self.df = df 
        self.train_mode = train_mode
        
        # train / test split 
        self.img_dirs = self.df[self.df['train/test'] == train_mode][0].values # column 0 : img_dirs 
        self.labels = self.df[self.df['train/test'] == train_mode]['anomaly'].values 
        self.masks = self.df[self.df['train/test'] == train_mode]['mask'].values 
        
        # ground truth 
        self.gt = gt # mode 
        self.gt_transform = gt_transform 

        # Image 
        self.transform = transform         
        self.name = 'VISA'        
        self.idx = idx 
        
    def _get_ground_truth(self,img_dir, img, idx):
        if self.labels[idx] == 1:
            img_dir = self.masks[idx]
            gt = Image.open(img_dir)
            gt = self.gt_transform(gt)
        else:
            # image = np.zeros_like(torch.permute(img,dims=(1,2,0))).astype(np.uint8)
            gt = torch.zeros([1, *img.size()[1:]])
        # gt = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gt
        
    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self,idx):
        img_dir = self.img_dirs[idx]
        # img = cv2.imread(img_dir)
        img = Image.open(img_dir).convert('RGB')     
        img = self.transform(img)
        img = img.type(torch.float32)
        label = self.labels[idx]
        
        if self.gt:
            gt = self._get_ground_truth(img_dir, img, idx)
            #gt = self.gt_transform(gt)
            gt = (gt > 0).float()
            #gt = gt.type(torch.int64)
            
            
            if self.idx:
                return img, label, gt, idx
            else:
                return img, label, gt
        
        else:
            return img, label 
        
        
