from typing import Tuple
from tqdm import tqdm

import torch
from torch import tensor
from torch.utils.data import DataLoader
import timm

import numpy as np
from sklearn.metrics import roc_auc_score

from .utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params            

class KNNExtractor(torch.nn.Module):
	def __init__(self,backbone, out_indices = None ,pool_last = False):
		super().__init__()

		self.feature_extractor = timm.create_model(
			backbone,
			out_indices=out_indices,
			features_only=True,
			pretrained=True,
		)
		for param in self.feature_extractor.parameters():
			param.requires_grad = False
		self.feature_extractor.eval()
		
		self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
		self.backbone_name = backbone # for results metadata
		self.out_indices = out_indices

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.feature_extractor = self.feature_extractor.to(self.device)
			
	def __call__(self, x: tensor):
		with torch.no_grad():
			feature_maps = self.feature_extractor(x.to(self.device))
		feature_maps = [fmap.to("cpu") for fmap in feature_maps]
		if self.pool:
			# spit into fmaps and z
			return feature_maps[:-1], self.pool(feature_maps[-1])
		else:
			return feature_maps

class SPADE(KNNExtractor):
	def __init__(self, k: int = 5, backbone: str = "resnet18"):
		super().__init__(
			backbone=backbone,
			out_indices=(1,2,3,-1),
			pool_last=True,
		)
		self.k = k
		self.image_size = 224
		self.z_lib = []
		self.feature_maps = []
		self.threshold_z = None
		self.threshold_fmaps = None
		self.blur = GaussianBlur(4)

	def fit(self, train_dl):
		for sample, _, _ in tqdm(train_dl, **get_tqdm_params()):
			feature_maps, z = self(sample)

			# z vector
			self.z_lib.append(z)

			# feature maps
			if len(self.feature_maps) == 0:
				for fmap in feature_maps:
					self.feature_maps.append([fmap])
			else:
				for idx, fmap in enumerate(feature_maps):
					self.feature_maps[idx].append(fmap)

		self.z_lib = torch.vstack(self.z_lib)
		
		for idx, fmap in enumerate(self.feature_maps):
			self.feature_maps[idx] = torch.vstack(fmap)

	def predict(self, sample):
		feature_maps, z = self(sample)

		distances = compute_distance_matrix(z, self.z_lib.squeeze())
		values, indices = torch.topk(distances.squeeze(), self.k, largest=False)
		
		z_score = values.mean(-1).reshape(-1,1)

		# Build the feature gallery out of the k nearest neighbours.
		# The authors migh have concatenated all features maps first, then check the minimum norm per pixel.
		# Here, we check for the minimum norm first, then concatenate (sum) in the final layer.
		scaled_s_map = torch.zeros(sample.shape[0],1,self.image_size,self.image_size)
		for idx, fmap in enumerate(feature_maps):
			nearest_fmaps = self.feature_maps[idx][indices]
			# min() because kappa=1 in the paper
			s_map, _ = torch.min(((nearest_fmaps - fmap.unsqueeze(1))**2).sum(2), dim=1)
			smap =  torch.nn.functional.interpolate(
				s_map.unsqueeze(1), size=(self.image_size,self.image_size), mode='bilinear'
			)
			scaled_s_map += smap 
		
		return z_score, scaled_s_map

	def get_parameters(self):
		return super().get_parameters({
			"k": self.k,
		})
    
  
  
def compute_distance_matrix(A, B):
    # A의 모든 행에 대해 반복
    D = torch.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            # 유클리드 거리 계산
            D[i, j] = torch.linalg.norm(A[i] - B[j])
    return D