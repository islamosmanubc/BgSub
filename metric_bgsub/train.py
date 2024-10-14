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

from dataset.BSB import BSBdataset
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

parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='BSB', choices=['BSB'])
parser.add_argument('--img-size', default=224, type=int)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--bs', default=8, type=int)
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
        trainset = BSBdataset('../../background_subtraction_dataset/Processed', size=size)
    else:
        raise NotImplementedError
    trainsampler = torch.utils.data.SequentialSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler,shuffle=False)
    
    criterion = FgSegLoss().cuda(local_rank)
    
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    total_iters = args.epochs * len(trainloader)

    #training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, bg, gt = sample[0].cuda(), sample[1].cuda(), sample[2].cuda()
            
            pred = model(img,bg)
            
            gt = torch.reshape(gt, (len(gt),1,224,224))
            loss = criterion(pred, gt)
            
            #wandb.log({"loss": loss})
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
            
            if rank == 0 and i % 100 == 0:
                logger.info('Epoch: {}/{}, Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(epoch, args.epochs, i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
            if i % 1000 == 0:
                if i == 0:
                    store_samples(img,pred,bg,gt,i+epoch)
                else:
                    store_samples(img,pred,bg,gt,i)

        if rank == 0: #and epoch % 5 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))


if __name__ == '__main__':
    main()
    #wandb.finish()