import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
from eval_functions import *


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice


class SegDataset(Dataset):
    def __init__(self,root_dir,annot_csv, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            annot_csv (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.annots = pd.read_csv(annot_csv)
        self.transform = transform
    
    def __len__(self):
        return len(self.annots)

    def __getitem__(self,inx):
        if torch.is_tensor(inx):
            inx = inx.tolist()

        # Retrieve meta data from annotations
        if isinstance(inx,list):
            sample = self.annots.loc[inx,:].to_dict('index')
        else:
            sample = self.annots.loc[inx,:].to_dict()

        # Decode image mask from RLE
        rle = self.annots.loc[inx,"rle"].split()
        rle = np.array([int(x) for x in rle]).reshape(-1,2)
        mask = DecodeRLE(rle,(sample['img_height'],sample['img_width']))
        # mask = np.dstack(((np.abs(np.ones(mask.shape)-mask)[:,:,None],mask[:,:,None])))


        # Load image
        img_name = os.path.join(self.root_dir, str(self.annots.iloc[inx,0])+'.tiff')
        # image = io.imread(img_name)
        image = Image.open(img_name)
        # print('Image type:',type(image))

        # Transform data if applicable
        if self.transform:
            # print(self.transform)
            image,mask = self.transform(image,mask)
        
        # mask = torch.from_numpy(mask)
        
        return  image, mask, sample

