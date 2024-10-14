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
from visualize import store_samples
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
parser.add_argument('--bs', default=8, type=int)
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
        trainset = BSBdataset('D:/0 phd/my_papers/paper 14 (SA)/background_subtraction_dataset/Processed', size=size)
    else:
        raise NotImplementedError
    trainsampler = torch.utils.data.SequentialSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler,shuffle=False)


    #training loop
    ths = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    model.eval()
    total_loss = 0
    fscores = {}
    for th in ths:
        fscores.update({th:[]})
    for i, sample in enumerate(trainloader):
        
        img, bg, gt = sample[0].cuda(), sample[1].cuda(), sample[2].cuda()
        
        pred = model(img,bg)
        
        gt = np.reshape(gt.detach().cpu().numpy(), (len(gt),1,224,224))
        out = np.zeros((len(gt),1,224,224))
        pred = pred.detach().cpu().numpy()
        for th in ths:
            out[pred>th] = 1
            out[pred<=th] = 0
            try:
                f1 = f1_score(out.flatten(), gt.flatten())
                fscores[th].append(f1)
            except:
                gg = 1
        if rank == 0:
            logger.info('Iter: {}/{}, fscore: {:.3f}'.format(i, len(trainloader), f1))

        #store_samples(img,pred,bg,gt,i)

    for th in ths:
        th_scores = fscores[th]
        th_scores = np.array(th_scores)
        avg_fm = np.mean(th_scores)
        print('at threshold {:.2f} the f-score is {:.3f}'.format(th,avg_fm))

if __name__ == '__main__':
    main()
    #wandb.finish()