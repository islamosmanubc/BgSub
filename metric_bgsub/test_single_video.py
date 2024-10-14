import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from dataset.BSBtest import BSBdataset
from Background_sub.bsb import BSUB
from util.dist_helper import setup_distributed
from util.loss import FgSegLoss
from util.utils import init_log
from visualize import store_all_samples

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from torchvision.transforms import Compose
#import wandb

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="BSUB-s",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 5e-6,
#     "train": "bce+iouloss",
#     "dataset": "bsub",
#     "epochs": 120,
#     "size": 224
#     }
# )
parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='BSB', choices=['BSB'])
parser.add_argument('--img-size', default=224, type=int)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--bs', default=32, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from',default='checkpoints/vits.pth', type=str)
parser.add_argument('--save-path',default='checkpoints', type=str)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    rank, world_size = setup_distributed(port=args.port)
    
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)

    # building model
    local_rank = 0
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = BSUB(**{**model_configs[args.encoder]})
    model.cuda(local_rank)
    

    checkpoint = torch.load(args.pretrained_from, map_location='cpu')['model']
    model.load_state_dict(checkpoint,strict=True)
    final_transform = Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])
    #training loop
    model.eval()
    video = 'sample_videos/frames/4'
    frames = os.listdir(video)
    back = os.listdir(video+'/back')
    back= back[0]
    back = os.path.join(video+'/back',back)
    bgg = Image.open(back)
    bgg = bgg.resize((224,224))
    bgg = final_transform(bgg).float().cuda()
    #bgg = np.reshape(np.array(bgg), (1,224,224,3))
    i = 0
    for frame in frames:
        if 'back' in frame:
            continue
        ff = os.path.join(video,frame)
        img = Image.open(ff)
        img = img.resize((224,224))
        img = final_transform(img).float().cuda()
        bg = torch.reshape(bgg,(1,3,224,224))
        img = torch.reshape(img,(1,3,224,224))
        pred = model(img,bg)
        store_all_samples(img,pred,bg,i)
        i+=1

if __name__ == '__main__':
    main()