import matplotlib.pyplot as plt 
import torch 
import numpy as np 

def img_show(tensor):
    tensor = img_cvt(tensor)
    tensor = normalizing(tensor)
    plt.imshow(tensor)
    plt.show()
    
    
def img_cvt(img):
    img = torch.permute(img, dims=(1,2,0)).detach().cpu().numpy()
    return img 

def normalizing(input):
    output = (input - np.min(input)) / (np.max(input) - np.min(input))
    return output 

def scattering(data, **option):
    if option:        
        plt.scatter(
            data[:, 0],
            data[:, 1],
            **option
        )
    else:
        plt.scatter(
            data[:, 0],
            data[:, 1]        
        )