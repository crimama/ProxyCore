import pandas as pd 
import numpy as np 
from typing import  Callable, Optional

import torch 
from torchvision import datasets 

class CIFAR10(datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        class_name: str,
        anomaly_ratio: float,
        baseline: bool = True,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        ) -> None:
        super(CIFAR10, self).__init__(
            root = root,
            train = train,
            transform = transform,
            target_transform = target_transform,
            download = download 
        )
        
        self.baseline = baseline 
        self.class_name = class_name 
        self.anomaly_ratio = anomaly_ratio
        self.name = 'CIFAR10'
        
        if self.baseline:
            self.set_base(class_name)            
        else:
            self.set_fully_unsu(anomaly_ratio, class_name)
            
    def __len__(self):
        return len(self.data)    
    
        
    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.transform(img)
        
        gt = torch.zeros_like(img)
        
        label = self.targets[idx]
        
        return img, label, gt
            
    def set_fully_unsu(self, anomaly_ratio, class_name):
        if self.train:
            num_train = 5000 # 50000 * 0.1 
            num_anomaly_train = int(num_train * anomaly_ratio)
            
            # anomaly 데이터 중 train에 포함 시킬 데이터 인덱스 sampling 
            index_anomaly_train = pd.Series(self.targets)[pd.Series(self.targets) != class_name].sample(num_anomaly_train).index
            
            # 전체 train 데이터 중 class에 해당하는 데이터만 샘플링 + 이상치를 추가할 만큼 제외 한 뒤 샘플링
            index_normal_train = pd.Series(np.where(np.array(self.targets) == class_name)[0]).sample(num_train - num_anomaly_train).values
            
            index_train = np.concatenate([index_normal_train, index_anomaly_train])
            self.data = self.data[index_train]
            self.targets = np.array(self.targets)[index_train]
            self.targets = pd.Series(self.targets).apply(lambda x : 0 if x == class_name else 1 ).values 
        else:
            self.targets = pd.Series(self.targets).apply(lambda x : 0 if x == class_name else 1 ).values 
                            
                                                
    def set_base(self, class_name):
        if self.train:
            self.data = self.data[np.where(np.array(self.targets) == class_name)[0]]
            self.targets = np.array(self.targets)[np.where(np.array(self.targets) == class_name)[0]]     
            self.targets = pd.Series(self.targets).apply(lambda x : 0 if x == class_name else 1 ).values     
        else:
            self.targets = pd.Series(self.targets).apply(lambda x : 0 if x == class_name else 1 ).values        
            
        
        