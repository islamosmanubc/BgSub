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
        self.all_gt = []
        self.all_bg = []
        self.subdataset_frames = {}
        self.subdataset_gt = {}
        self.subdataset = []
        self.videos = []
        for filename in os.listdir(dataset_path):
            if filename == 'DAVIS2016' or filename == 'PTZ' or filename == 'SegTrackV2':
                continue
            self.subdataset.append(filename)
            self.subdataset_frames.update({filename:[]})
            self.subdataset_gt.update({filename:[]})
            gg = os.path.join(dataset_path,filename)
            for vid in os.listdir(gg):
                ff = os.path.join(gg, vid)
                if not os.path.isdir(ff):
                    continue
                
                gt = []
                for frame in os.listdir(ff+'/groundtruth'):
                    if 'png' in frame:
                        im = Image.open(os.path.join(ff+'/groundtruth',frame))
                        inp = np.array(im)
                        blackpixels = inp[inp == 0]
                        if len(blackpixels)> 50: 
                            self.subdataset_gt[filename].append(ff+'/groundtruth/'+frame)
                            self.all_gt.append(ff+'/groundtruth/'+frame)
                            gt.append(frame)
                    
                for frame in os.listdir(ff+'/input'):
                    if frame.replace('in','gt').replace('jpg','png') in gt:
                        self.subdataset_frames[filename].append(frame)
                        self.all_frames.append(ff+'/input/'+frame)
                        self.subdataset.append(filename)
                        self.all_bg.append(ff+'/back/back.jpg')
                        self.videos.append(vid)
                #break
        
        

        self.final_transform = Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])
        
    def __getitem__(self, idx):
        frame = self.all_frames[idx]
        gt = self.all_gt[idx]
        background = self.all_bg[idx]
        subdataset = self.subdataset[idx]
        vid = self.videos[idx]
        this_im = Image.open(frame).convert('RGB')
        this_im = this_im.resize((224,224))
        this_bg = Image.open(background).convert('RGB')
        this_bg = this_bg.resize((224,224))
        this_gt = Image.open(gt)
        this_gt = this_gt.resize((224,224))
        inp = np.array(this_gt)
        inp[inp>200] = 255
        inp[inp<=200] = 0
        this_gt = Image.fromarray(np.uint8(inp))
        
        this_im = self.final_transform(this_im).float()
        this_bg = self.final_transform(this_bg).float()
        this_gt = np.asarray(np.array(this_gt)/255.0,dtype=np.float32)

        return this_im,this_bg,this_gt,subdataset,vid

    def __len__(self):
        return len(self.all_frames)