from copy import deepcopy 
from scipy import stats

import torch 
import numpy as np 
from torch.utils.data import DataLoader, SubsetRandomSampler

from .sampler import SubsetSequentialSampler

class Refinementer:
    def __init__(self, model, n_query: int, dataset, unrefined_idx: np.ndarray, 
                 batch_size: int, test_transform, num_workers: int):
        
        self.model = model 
        self.n_query = n_query 
        self.dataset = dataset 
        self.unrefined_idx = unrefined_idx 
        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.test_transform = test_transform 
        
    
    def init_model(self):
        if self.model.__class__.__name__ == 'PatchCore':
            self.model.__init_model__()
            return self.model
        else:
            return deepcopy(self.model)    
    
    def update(self, query_idx: np.ndarray) -> DataLoader:
        '''
        Create new DataLoader getting new query 
        '''        
        self.unrefined_idx[query_idx] = False 
        
        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = self.batch_size,
            sampler     = SubsetRandomSampler(indices = np.where(self.unrefined_idx==True)[0]),
            num_workers = self.num_workers
        )
        return dataloader 
    
    def get_grad_embedding(self, model, unrefined_idx: np.ndarray) -> torch.Tensor:
        
        dataset = deepcopy(self.dataset)
        dataset.transform = self.test_transform 
        device = next(model.parameters()).device 
        
        dataloader = DataLoader(
            dataset     = dataset,
            batch_size  = self.batch_size, 
            sampler     = SubsetSequentialSampler(indices=unrefined_idx),
            num_workers = self.num_workers
        )
        
        grad_embeddings = []
        for imgs, _, _ in dataloader:
            output = model(imgs.to(device))
            loss = model.criterion(output)
            grads = torch.autograd.grad(loss, output[1])
            # grad_embeddings.append(
            #     torch.hstack([torch.norm(g.mean(1),dim=(1,2)).unsqueeze(1) for g in grads])
            # )
            grad_embeddings.append(
                torch.hstack([torch.norm(torch.norm(g,dim=1),dim=(1,2)) for g in grads])
            )
        grad_embeddings = torch.vstack(grad_embeddings)
        return grad_embeddings 
    
    def query(self, model):
        unrefined_idx = np.where(self.unrefined_idx==True)[0]
        
        grad_embeddings = self.get_grad_embedding(model, unrefined_idx)
        chosen = init_centers(X = grad_embeddings, K = self.n_query)
        return chosen 
    
    
                                             
                                        
@torch.no_grad()
def init_centers(X: torch.Tensor, K: int = None):    
    '''
    K-MEANS++ seeding algorithm 
    - 초기 센터는 gradient embedding의 norm이 가장 큰 것으로 사용 
    - 그 이후 부터는 앞서 선택 된 center와의 거리를 계산, 이와 비례하는 확률로 이산 확률 분포를 만듬
    - 이 이산 확률 분포에서 새로운 센터를 선택 
    - 이렇게 뽑힌 센터들은 k-means 목적함수의 기대값에 근사되는 것이 증명 됨, 따라서 다양성을 확보할 수 있음 (Arthur and Vassilvitskii, 2007)
    '''    
    # K-means ++ initializing
    embs = torch.Tensor(X)
    ind = torch.argmax(torch.norm(embs, 2, 1)).item()
    embs = embs.cuda()
    
    mu = [embs[ind]]
    indsAll = [ind]
    centInds = [0.] * len(embs)
    cent = 0
    
    # Sampling 
    while len(mu) < K:
        if len(mu) == 1:
            D2 = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy() # Calculate l2 Distance btw mu and embs 
        else:
            newD = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy() # Calculate l2 Distance btw mu and embs 
            for i in range(len(embs)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]            
        D2 = D2.ravel().astype(float)
        
        Ddist = (D2 ** 2)/ sum(D2 ** 2) 
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist)) # 이산 확률 분포 구축 
        ind = customDist.rvs(size=1)[0] # 이산 확률 분포에서 하나 추출 
        
        while ind in indsAll: ind = customDist.rvs(size=1)[0] # repeat until choosing index not in indsAll 
        
        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll                                        