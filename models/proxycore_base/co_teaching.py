import torch 
import torch.nn as nn 
import torch.nn.functional as F 
# from criterions.proxynca import ProxyNCA, proxy_anchor_loss, NormSoftmaxLoss
import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from .layers import MLP,FeatureDataset
from .model import ProxyCoreBase
from criterions import NormSoftmaxLoss

class CoTeaching(ProxyCoreBase):
    def __init__(self, backbone, faiss_on_gpu, faiss_num_workers, 
                 sampling_ratio, device, input_shape, threshold='quant15', weight_method='identity',
                 n_input_feat:int=1024, n_hidden_feat:int=4096, n_projection_feat:int=1024
                ):
        super(CoTeaching,self).__init__(self, backbone, faiss_on_gpu, faiss_num_workers, 
                 sampling_ratio, device, input_shape, threshold=threshold, weight_method=weight_method,
                 n_input_feat=n_input_feat, n_hidden_feat=n_hidden_feat, n_projection_feat=n_projection_feat
                )    
        
        self.embedding_layer_2 = MLP(n_input_feat,n_hidden_feat,n_projection_feat)
        self.projection_layer_2 = MLP(n_projection_feat,n_hidden_feat,n_projection_feat)
                    
    
    def criterion(self, outputs: torch.Tensor, labels: torch.Tensor, epoch: int):
        return self._criterion(outputs, labels, epoch)    
    
    def forward(self, feat):
        output_1 = self.projection_layer(self.embedding_layer(feat))   
        output_2 = self.projection_layer_2(self.embedding_layer_2(feat))  
        
        return output_1, output_2 
        
        
class CoTeachingLoss(nn.Module):
    def __init__(self, sz_embedding = 1024, nb_classes=784,
                 forget_rate=0.2, num_gradual= 1 , exponent = 1):
        super(CoTeachingLoss,self).__init__()        
        
                
        self.rate_schedule = np.ones(200) * forget_rate 
        self.rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)
        
        self._criterion = NormSoftmaxLoss(
            nb_classes = nb_classes,
            sz_embedding = sz_embedding
        )
        
    def forward(self, outputs:list, labels, epoch):
        '''
            outputs = [output1, output2]
        '''
        forget_rate = self.rate_schedule[epoch]
        
        logits1, logits2 = outputs                 
        [pred1,pred2] = [torch.max(l,dim=1)[1] for l in outputs]                
        
        inds = torch.where(pred1 != pred2)  # 튜플에서 첫 번째 값(인덱스들)만 선택              
        if len(inds[0]) * (1 - self.rate_schedule[epoch]) < 1:
            loss_1 = self._criterion(logits1, labels)
            loss_2 = self._criterion(logits2, labels)
            return loss_1, loss_2
        else:
            # To select instances have Lower loss 
            #with torch.no_grad():
            with torch.no_grad():
                loss_1 = self._criterion(logits1, labels, mean=False)
                ind_1_sorted = np.argsort(loss_1.data.cpu())            
                
                loss_2 = self._criterion(logits2, labels)
                ind_2_sorted = np.argsort(loss_2.data.cpu())
                
                remember_rate = 1 - forget_rate 
                num_remember = int(remember_rate * len(ind_1_sorted))
                
            ind_1_update = ind_1_sorted[:num_remember]
            ind_2_update = ind_2_sorted[:num_remember]                        
            
            # loss for update 
            loss_1_update = self._criterion(logits1[ind_2_update], labels[ind_2_update])
            loss_2_update = self._criterion(logits2[ind_1_update], labels[ind_1_update])
            
            return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update/num_remember)
    