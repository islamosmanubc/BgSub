import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import random

def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

class BSBdataset(Dataset):
    def __init__(self, dataset_path, size=(224, 224)):
        
        self.size = size
        self.all_frames = []
        for subdataset in os.listdir(dataset_path):
            subdataset_path = os.path.join(dataset_path,subdataset)
            if 'CDNet' not in subdataset:
                continue
            for video in os.listdir(subdataset_path):
                video_path = os.path.join(subdataset_path,video)
                for frame in os.listdir(video_path+'/Frames'):
                    self.all_frames.append(os.path.join(video_path,'Frames',frame))
            #break
        
        
        self.all_frames = np.asarray(self.all_frames)
        #self.shuffle()

        self.final_transform = Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])
        
    def shuffle(self):
        idx = list(range(self.all_frames.shape[0]))
        np.random.shuffle(idx)
        np.random.shuffle(idx)
        self.all_frames = self.all_frames[idx]

    def __getitem__(self, idx):
        frame = self.all_frames[idx]
        gt = frame.replace('Frames','Annotations')
        background = frame.replace('Frames','Background')
        this_im = Image.open(frame).convert('RGB')
        #this_im = self.frame_transform(this_im)
        this_bg = Image.open(background).convert('RGB')
        #this_bg = self.background_transform(this_bg)
        this_gt = Image.open(gt).convert('P')
        
        #seed = np.random.randint(2147483647)

        #reseed(seed)
        #this_im = self.all_transform(this_im)
        #reseed(seed)
        #this_gt = self.all_transform(this_gt)
        #reseed(seed)
        #this_bg = self.all_transform(this_bg)

        #seed = np.random.randint(2147483647)
        
        #reseed(seed)
        #this_im = self.frame_aff_transform(this_im)
        #reseed(seed)
        #this_gt = self.gt_aff_transform(this_gt)
        #reseed(seed)
        #this_bg = self.frame_aff_transform(this_bg)

        this_im = self.final_transform(this_im).float()
        this_bg = self.final_transform(this_bg).float()
        this_gt = np.asarray(np.array(this_gt)/255.0,dtype=np.float32)
        #this_gt = np.array(this_gt)/255.0

        return this_im,this_bg,this_gt

    def __len__(self):
        return len(self.all_frames)