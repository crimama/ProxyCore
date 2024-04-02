import numpy as np
import torch
from torch.utils.data import Sampler

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices: np.ndarray):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)