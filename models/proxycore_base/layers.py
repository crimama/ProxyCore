import torch
import torch.nn as nn 

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
    
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features 
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        x = self.features[idx]
        y = idx - (784 * (idx//784))
        return x,y 