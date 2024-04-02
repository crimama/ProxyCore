import torch 
import torch.nn as nn 
import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from models import PatchCore
from criterions import NormSoftmaxLoss
from .layers import MLP, FeatureDataset


'''
추후 Feature dataset과 MLP, proxy 등 나눠서 리팩토링 필요 
Featuredataset -> dataset으로 이동 
MLP Encoder class 다시 만들고 
ProxyBase | Coteaching 이런 식으로 추가 
어찌되었던 전반적인 리팩토링 필요 

장기적으로 patch + learning 방식들이랑 기존 베이스라인 프레임워크 분리해야 할듯 
분리하는 과정에서 refinement 없애기 
'''

class ProxyCoreBase(nn.Module):
    def __init__(self, backbone, faiss_on_gpu, faiss_num_workers, 
                 sampling_ratio, device, input_shape, threshold='quant15', weight_method='identity',
                 n_input_feat:int=1024, n_hidden_feat:int=4096, n_projection_feat:int=1024
                ):
        super(ProxyCoreBase,self).__init__()
        
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
        
        self._criterion = NormSoftmaxLoss(
            nb_classes = 784,
            sz_embedding = n_projection_feat
        )            
        self.device = device
        
    def criterion(self, outputs:list):
        '''
            outputs = [z,w]
        '''
        return self._criterion(*outputs)
    
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
    
    def get_feature_loader(self, trainloader):
        features = self.get_patch_embed(trainloader)
        featuredataset = FeatureDataset(features)
        featureloader = DataLoader(featuredataset,batch_size=512, shuffle=True)
        return featureloader
        
    def forward(self, feat):     
        output = self.projection_layer(self.embedding_layer(feat))   
        return output   
        
