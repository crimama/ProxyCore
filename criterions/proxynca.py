import math
import numpy as np     
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0.1):
    # Optional: BNInception uses label smoothing, apply it for retraining also
    # "Rethinking the Inception Architecture for Computer Vision", p. 6
    T = torch.eye(nb_classes).to(T.device)[T]
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    return T


class ProxyNCA(torch.nn.Module):
    def __init__(self, 
        nb_classes,
        sz_embedding,
        smoothing_const = 0.1,
        scaling_x = 1,
        scaling_p = 3
    ):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.proxies = Parameter(torch.randn(nb_classes, sz_embedding) / 8)
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p

    def forward(self, X, T, mean=True):
        '''
        learning using initial position based fixed label 
        '''
        P = F.normalize(self.proxies, p = 2, dim = -1) * self.scaling_p
        X = F.normalize(X, p = 2, dim = -1) * self.scaling_x
        D = torch.cdist(X, P) ** 2
        T = binarize_and_smooth_labels(T, len(P), self.smoothing_const)
        # note that compared to proxy nca, positive included in denominator
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        if mean:
            return loss.mean()
        else:
            return loss
    
    def get_p_label(self, X):
        '''
        learning using pseudo label based on similarity btw proxies and embed 
        '''
        P = F.normalize(self.proxies, p = 2, dim = -1) * self.scaling_p
        X = F.normalize(X, p = 2, dim = -1) * self.scaling_x
        D = torch.cdist(X, P) ** 2
        T = F.normalize(D, p=2, dim=-1)
        T = T.argmax(dim=1)
        return T
    
    
    
class ProxyAnchor(torch.nn.Module):
    '''
    ref: https://arxiv.org/abs/2003.13911
    official pytorch codes: https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    '''
    def __init__(self, 
                 nb_classes,
                 sz_embedding,
                 scale = 32, 
                 margin = 0.1
                 ):
        super(ProxyAnchor, self).__init__()
        self.proxy = Parameter(torch.Tensor(sz_embedding, nb_classes))
        self.n_classes = nb_classes
        self.alpha = scale
        self.delta = margin
        torch.nn.init.kaiming_normal_(self.proxy, mode='fan_out')

    def forward(self, embeddings, target):
        embeddings_l2 = F.normalize(embeddings, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=0)
        
        # N, dim, cls

        sim_mat = embeddings_l2.matmul(proxy_l2) # (N, cls)
        
        pos_target = F.one_hot(target, self.n_classes).float()
        neg_target = 1.0 - pos_target
        
        pos_mat = torch.exp(-self.alpha * (sim_mat - self.delta)) * pos_target
        neg_mat = torch.exp(self.alpha * (sim_mat + self.delta)) * neg_target
        
        pos_term = 1.0 / torch.unique(target).shape[0] * torch.sum(torch.log(1.0 + torch.sum(pos_mat, axis=0)))
        neg_term = 1.0 / self.n_classes * torch.sum(torch.log(1.0 + torch.sum(neg_mat, axis=0)))

        loss = pos_term + neg_term

        return loss
    


class SoftTriple(nn.Module):
    def __init__(self, la=20, gamma=0.1, tau=0.2, margin=0.01, dim=1024, cN=28, K=28):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify
        
        
class NormSoftmaxLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    https://github.com/azgo14/classification_metric_learning/blob/master/metric_learning/modules/losses.py
    """
    def __init__(self,
                 sz_embedding,
                 nb_classes,
                 temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()

        self.weight = Parameter(torch.Tensor(nb_classes, sz_embedding))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets, mean=True):
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)
        
        loss = nn.CrossEntropyLoss(reduction = 'mean' if mean == True else 'none')(prediction_logits / self.temperature, instance_targets)
        return loss 


class CoreProxy(nn.Module):
    def __init__(self,
                 init_core :np.ndarray,
                 temperature=0.05,
                 loss_fn = 'CrossEntropy'):
        '''
        default tempearture : 0.05
        '''
        super(CoreProxy, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(init_core))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = loss_fn

    def forward(self, embeddings, instance_targets, reduction:bool=True) -> torch.Tensor:
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)
        
        if self.loss_fn == 'CrossEntropy':
            loss = nn.CrossEntropyLoss(reduction = 'mean' if reduction == True else 'none')(prediction_logits / self.temperature, instance_targets)
        elif self.loss_fn == 'focalloss':
            loss = FocalLoss(reduce = reduction)(prediction_logits / self.temperature, instance_targets)
        else:
            raise NotImplementedError
        
        
        # loss = FocalLoss(reduce = False)(prediction_logits / self.temperature, instance_targets)
        
        # ### noise score ### 
        # from scipy.special import lambertw
        # from skimage.filters import threshold_otsu
        
        # lam = 0.5 
        # tau = threshold_otsu(loss.detach().cpu().numpy())   

        # inner_term = (loss.detach().cpu()-tau) / (2 * lam)
        # inner_term_pos = torch.relu(inner_term)
        # w_value = lambertw(inner_term_pos)
        # sigma = torch.exp(-w_value)
        # sigma = sigma.real.type(torch.float32)
        
        # loss = torch.mean(sigma.to(loss.device) * loss )
        
        return loss 
    
    
    
import torch.nn as nn 
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
    
        BCE_loss = nn.CrossEntropyLoss(reduction = 'mean' if self.reduce ==True else 'none')(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss