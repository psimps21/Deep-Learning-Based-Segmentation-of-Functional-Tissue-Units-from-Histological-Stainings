import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
from eval_functions import *


class SegDataset(Dataset):
    def __init__(self,root_dir,annot_csv, tissue, transform=None, augment=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            annot_csv (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        annot = pd.read_csv(annot_csv)
        self.annots = annot.loc[annot['organ']==tissue]
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.annots)

    def __getitem__(self,inx):
        if torch.is_tensor(inx):
            inx = inx.tolist()

        # Retrieve meta data from annotations
        sample = self.annots.iloc[inx,:].to_dict()

        # Decode image mask from RLE
        rle = self.annots.iloc[inx,:]["rle"].split()
        rle = np.array([int(x) for x in rle]).reshape(-1,2)
        mask = DecodeRLE(rle,(sample['img_height'],sample['img_width']))

        # Load image
        img_name = os.path.join(self.root_dir, str(self.annots.iloc[inx,0])+'.tiff')
        image = Image.open(img_name)

        # Transform data if applicable
        if self.transform:
            image,mask = self.transform(image,mask)
            
        if self.augment:
            image, mask = self.augment(image,mask)
        #image_more, mask_more = torchvision.transforms.RandomHorizontalFlip(0.5),
            #append image more to image
        #self.augment
                
        return  image, mask
