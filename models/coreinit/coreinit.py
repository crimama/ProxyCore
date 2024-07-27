import torch 
import torch.nn as nn 
import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from models import PatchCore
from criterions import NormSoftmaxLoss
from models.proxycore_base.layers import MLP, FeatureDataset

import math 
import torch.nn as nn 
from criterions import CoreProxy, ProxyNCA 

    
class CoreInit(nn.Module):
    '''
    pslabel_sampling_ratio 는 fit에서 초기 pseudo label 생성을 위한 coreset을 위한 용도
    당장 여기서 사용 되지는 않지만 config yaml파일에서 params.로 넣기 위해 임시로 넣어둠 
    '''
    def __init__(self, backbone, faiss_on_gpu, faiss_num_workers, pslabel_sampling_ratio,
                 sampling_ratio, device, input_shape, threshold='quant_0.15', weight_method='identity',
                 n_input_feat:int=1024, n_hidden_feat:int=4096, n_projection_feat:int=1024,
                 temperature=0.05, loss_fn='CrossEntropy'
                ):
        super(CoreInit,self).__init__()
        self.core = PatchCore(
            backbone          = backbone,
            faiss_on_gpu      = faiss_on_gpu,
            faiss_num_workers = faiss_num_workers,
            sampling_ratio    = sampling_ratio,
            device            = device,
            input_shape       = input_shape,
            threshold         = threshold,
            weight_method     = weight_method
            
        )
        
        self.embedding_layer = MLP(n_input_feat,n_hidden_feat,n_projection_feat)
        self.projection_layer = MLP(n_projection_feat,n_hidden_feat,n_projection_feat)
        
        
        self.device = device
        self.temperature = temperature
        self.loss_fn = loss_fn
        
    def set_criterion(self, init_core):
        self._criterion = CoreProxy(
           init_core   = init_core,
           temperature = self.temperature,
           loss_fn     = self.loss_fn
        )
        
    # def set_criterion(self, init_core):
    #     self._criterion = ProxyNCA(
    #         *init_core.shape
    #     )
        
    def criterion(self, outputs:list,reduction='mean'):
        '''
            outputs = [z,w]
        '''
        return self._criterion(*outputs,reduction=reduction)
    
    def get_patch_embed(self, trainloader):
        self.core.forward_modules.eval()
        
        features = [] 
        for imgs, labels, gts in trainloader:
            imgs = imgs.to(torch.float).to(self.device)
            with torch.no_grad():
                feature = self.core._embed(imgs) # (N*P*P,C)
                feature = torch.Tensor(np.vstack(feature))
                features.append(feature)        
            
        origin_patch_embeds = torch.concat(features)
        return origin_patch_embeds

    def get_feature_loader(self, trainloader, labels=None):
        features = self.get_patch_embed(trainloader)
        featuredataset = FeatureDataset(features, labels=labels)
        featureloader = DataLoader(featuredataset,batch_size=512, shuffle=True)
        return featureloader
        
    def forward(self, feat):     
        output = self.projection_layer(self.embedding_layer(feat))   
        return output   
        
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels=None):
        self.features = features 
        self.labels = labels 
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        x = self.features[idx]
        y = idx - (784 * (idx//784)) if self.labels is None else self.labels[idx]        
        return x,y 