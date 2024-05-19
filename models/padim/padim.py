from tqdm import tqdm

import torch
from torch import tensor
import timm

import sys 

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

TQDM_PARAMS = {
	"file" : sys.stdout,
	"bar_format" : "   {l_bar}{bar:10}{r_bar}{bar:-10b}",
}

def get_tqdm_params():
    return TQDM_PARAMS

class PaDiM(KNNExtractor):
	def __init__(
		self,
		d_reduced: int = 100,
		backbone: str = "resnet18",
	):
		super().__init__(
			backbone=backbone,
			out_indices=(1,2,3),
		)
		self.image_size = 224
		self.d_reduced = d_reduced # your RAM will thank you
		self.epsilon = 0.04 # cov regularization
		self.patch_lib = []
		self.resize = None

	def fit(self, train_dl):
		for sample, _, _  in tqdm(train_dl, **get_tqdm_params()):
			feature_maps = self(sample)
			if self.resize is None:
				largest_fmap_size = feature_maps[0].shape[-2:]
				self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
			resized_maps = [self.resize(fmap) for fmap in feature_maps]
			self.patch_lib.append(torch.cat(resized_maps, 1))
		self.patch_lib = torch.cat(self.patch_lib, 0)

		# random projection
		if self.patch_lib.shape[1] > self.d_reduced:
			print(f"   PaDiM: (randomly) reducing {self.patch_lib.shape[1]} dimensions to {self.d_reduced}.")
			self.r_indices = torch.randperm(self.patch_lib.shape[1])[:self.d_reduced]
			self.patch_lib_reduced = self.patch_lib[:,self.r_indices,...]
		else:
			print("   PaDiM: d_reduced is higher than the actual number of dimensions, copying self.patch_lib ...")
			self.patch_lib_reduced = self.patch_lib

		# calcs
		self.means = torch.mean(self.patch_lib, dim=0, keepdim=True)
		self.means_reduced = self.means[:,self.r_indices,...]
		x_ = self.patch_lib_reduced - self.means_reduced

		# cov calc
		self.E = torch.einsum(
			'abkl,bckl->ackl',
			x_.permute([1,0,2,3]), # transpose first two dims
			x_,
		) * 1/(self.patch_lib.shape[0]-1)
		self.E += self.epsilon * torch.eye(self.d_reduced).unsqueeze(-1).unsqueeze(-1)
		self.E_inv = torch.linalg.inv(self.E.permute([2,3,0,1])).permute([2,3,0,1])

	def predict(self, sample):
		feature_maps = self(sample)
		resized_maps = [self.resize(fmap) for fmap in feature_maps]
		fmap = torch.cat(resized_maps, 1)

		# reduce
		x_ = fmap[:,self.r_indices,...] - self.means_reduced

		left = torch.einsum('abkl,bckl->ackl', x_, self.E_inv)
		s_map = torch.sqrt(torch.einsum('abkl,abkl->akl', left, x_))
		scaled_s_map = torch.nn.functional.interpolate(
			s_map.unsqueeze(0), size=(self.image_size,self.image_size), mode='bilinear'
		)

		return torch.Tensor([s.max() for s in s_map]), scaled_s_map[0, ...]

	def get_parameters(self):
		return super().get_parameters({
			"d_reduced": self.d_reduced,
			"epsilon": self.epsilon,
		})