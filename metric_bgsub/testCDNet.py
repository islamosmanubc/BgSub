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
from dataset.CDNettest import BSBdataset
from Background_sub.bsb import BSUB
from util.dist_helper import setup_distributed
from util.loss import FgSegLoss
from util.utils import init_log
from visualize import store_vid_samples
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

parser.add_argument('--encoder', default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='BSB', choices=['BSB'])
parser.add_argument('--img-size', default=224, type=int)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--bs', default=16, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from',default='checkpoints/vitb.pth', type=str)
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
    #x = torch.zeros((1,3,224,224))
    #y = torch.zeros((1,3,224,224))
    #x = x.cuda()
    #y = y.cuda()
    model.cuda(local_rank)
    #z = model(x,y)
    #if args.pretrained_from:
    #    model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    

    checkpoint = torch.load(args.pretrained_from, map_location='cpu')['model']
    model.load_state_dict(checkpoint,strict=True)
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    #dataset loading
    if args.dataset == 'BSB':
        trainset = BSBdataset('D:/0 phd/my_papers/paper 4 (REFNet)/codes/MOD4+edge/datasets/CDNet', size=size)
    else:
        raise NotImplementedError
    trainsampler = torch.utils.data.SequentialSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler,shuffle=False)


    #training loop
    model.eval()
    subdatasets = {}
    for i, sample in enumerate(trainloader):
        
        img, bg, gt,subd,vid = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(),sample[3],sample[4]
        if not subdatasets.__contains__(subd[0]):
            subdatasets.update({subd[0]:[]})
        pred = model(img,bg)
        
        gt = np.reshape(gt.detach().cpu().numpy(), (len(gt),1,224,224))
        out = np.zeros((len(gt),1,224,224))
        pred = pred.detach().cpu().numpy()
        #for th in ths:
        th = 0.5
        out[pred>th] = 1
        out[pred<=th] = 0
        for j in range(len(out)):
            o1 = out[j:j+1,:,:,:]
            g1 = gt[j:j+1,:,:,:]
            try:
                f1 = f1_score(o1.flatten(), g1.flatten())
                if f1 == 0:
                    subdatasets[subd[0]].append(1)
                else:
                    subdatasets[subd[0]].append(f1)
            except:
                subdatasets[subd[0]].append(1)
        if rank == 0:
            #logger.info('Iter: {}/{}, fscore: {:.3f}'.format(i, len(trainloader), f1))
            logger.info('Iter: {}/{}'.format(i, len(trainloader)))

        #store_vid_samples(img,pred,bg,gt,i,subd,vid)

    for subd,scores in subdatasets.items():
        scores = np.array(scores)
        avg_fm = np.mean(scores)
        print(subd+ ' f-score is {:.3f}'.format(avg_fm))

if __name__ == '__main__':
    main()