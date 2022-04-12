import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import cv2
from skimage import io



class FFTDataset(Dataset):
    
    def __init__(self, image, transform):

        if os.path.isfile(image) == False:
            raise Exception ("no iamge at this location")
        else:
            self.image = image
            self.transform = transform
        
    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        img = io.imread(self.image, as_gray = True)
        img = torch.FloatTensor(img[None, :, :])
        
        if self.transform is not None:
            transformed = self.transform(img)
            img = transformed
        
        sample = {"image": img}

        return sample