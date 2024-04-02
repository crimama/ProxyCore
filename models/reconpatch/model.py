import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
import timm 
import os 
from models import PatchCore


class ReconPatch(nn.Module):
    def __init__(self, backbone, faiss_on_gpu, faiss_num_workers, 
                 sampling_ratio, device, input_shape, threshold='quant15', weight_method='identity',
                 n_input_feat:int=1024, n_hidden_feat:int=4096, n_projection_feat:int=1024, k = 5, alpha = 0.5
                ):
        super(ReconPatch,self).__init__()
        
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

        self.ema_embedding_layer = MLP(n_input_feat,n_hidden_feat,n_projection_feat)
        self.ema_projection_layer = MLP(n_projection_feat,n_hidden_feat,n_projection_feat)
        
        self._criterion = RelaxedContrastiveLoss()
        
        self.pairwise_similarity= PairwiseSimilarity()
        self.contextual_similarity = ContextualSimilarity(k)
        self.alpha = alpha
        
        self.ema_updater = EMA(beta = 0.97)
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
        if self.training:
            with torch.no_grad():
                z = self.ema_projection_layer(self.embedding_layer(feat))
                p_sim = self.pairwise_similarity(z)
                c_sim = self.contextual_similarity(z)
                w = self.alpha * p_sim + (1-self.alpha) * c_sim
                
            z = self.projection_layer(self.embedding_layer(feat))            
            return [z,w]
                
    def ema_update(self):
        self.ema_updater.update_moving_average(self.ema_embedding_layer, self.embedding_layer)
        self.ema_updater.update_moving_average(self.ema_projection_layer, self.projection_layer)
        
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP,self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size 
        self._output_size = output_size 
        self.net = self.__net__(input_size, hidden_size, output_size)
        
    def __net__(self, input_size, hidden_size, output_size):
        return nn.Sequential(
            nn.Linear(in_features= input_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )
        
    def forward(self, inputs):
        return self.net(inputs)
    
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
    def update_moving_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)        
        
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features 
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        x = self.features[idx]
        y = idx - (784 * (idx//784))
        return x,y 
    
def l2_distance(z):
    diff = torch.unsqueeze(z,dim=1) - torch.unsqueeze(z,dim=0)
    return torch.sum(diff**2,dim=-1)

class PairwiseSimilarity():
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, z):
        return torch.exp(-l2_distance(z) / self.sigma)
    
class ContextualSimilarity():
    def __init__(self, k):
        self.k = k
        
    def __call__(self, z):
        self.k = 5 
        distances = l2_distance(z)
        kth_nearst = -torch.topk(-distances, k=self.k, sorted=True)[0][:, -1]
        mask = (distances <= torch.unsqueeze(kth_nearst,dim=-1)).type(torch.float32)
        
        similarity = torch.matmul(mask, mask.transpose(0, 1)) / torch.sum(mask, dim=-1, keepdim=True)
        R = mask * mask.T
        similarity = torch.matmul(similarity, R.transpose(0, 1)) / torch.sum(R, dim=-1, keepdim=True)
        return  0.5 * (similarity + similarity.T)
        
class RelaxedContrastiveLoss():
    def __init__(self, margin:float = 0.1):
        self.margin = margin 
        
    def __call__(self, z, w):
        '''
            z : projected feature 
            w : similarity calculated by ema network
        '''
        distances = torch.sqrt(l2_distance(z) + 1e-9)

        # delta 계산
        mean_distances = torch.mean(distances, dim=-1, keepdim=True)
        delta = distances / mean_distances

        # rc_loss 계산
        rc_loss = torch.sum(
            torch.mean(
                w * (delta ** 2) + (1 - w) * (torch.relu(self.margin - delta) ** 2),
                dim=-1
            )
        )
        return rc_loss 