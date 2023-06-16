from torch.utils.data import Dataset
import h5py
import torch
from PIL import Image
import numpy as np
    
class Shapes3D(Dataset):
    
    def __init__(self, root, phase,  transform=None,):
        
        with h5py.File(root, 'r') as f:
            self.imgs = f[phase][()]
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.imgs[index]
        num_relations, num_example, H, W, C = img.shape
        img = torch.from_numpy(img.reshape(-1, H, W, C))
        img = img.permute(0, -1, -3, -2)
        if self.transform is not None:
            img = self.transform(img)
        img = img.float() / 255.
        img_size = img.shape[-1]
        return img.reshape(num_relations, num_example, C, img_size, img_size)

    def __len__(self):
        return len(self.imgs)

class BitMoji(Dataset):

    def __init__(self, root, phase,  transform=None,):
       
        with h5py.File(root, 'r') as f:
            self.imgs = f[phase][()]
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.imgs[index]
        num_relations, num_example, H, W, C = img.shape
        img = torch.from_numpy(img.reshape(-1, H, W, C))
        img = img.permute(0, -1, -3, -2)
        if self.transform is not None:
            img = self.transform(img)
        img = img.float() / 255.
        img_size = img.shape[-1]
        return img.reshape(num_relations, num_example, C, img_size, img_size)

    def __len__(self):
        return len(self.imgs)

    
class CLEVr(Dataset):
    def __init__(self, root, phase,  transform=None,):
        
        with h5py.File(root, 'r') as f:
            self.imgs = f[phase][()]
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.imgs[index]
        num_relations, num_example, H, W, C = img.shape
        img = torch.from_numpy(img.reshape(-1, H, W, C))
        img = img.permute(0, -1, -3, -2)
        if self.transform is not None:
            img = self.transform(img)
        img = img.float() / 255.
        img_size = img.shape[-1]
        return img.reshape(num_relations, num_example, C, img_size, img_size)

    def __len__(self):
        return len(self.imgs)





    
