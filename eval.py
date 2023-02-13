import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import argparse
import numpy as np
import time
import logging
from utils.log import setup_logging

from utils.meters import AverageMeter, accuracy
from dataset import RFMCDataset
from models.resnet import ResNet_N_32x32, ResNet_S_32x32, BasicBlock, Bottleneck
from models.resnet1d import ResNet1d
from utils.preprocess import data_transofrm, csv2input, wav2input, IQwav2input, IQwav2input1d
from utils.reprocess import output2class, output2odd

parser = argparse.ArgumentParser()

parser.add_argument('--test_path', type=str, default='./audio_example.csv')
parser.add_argument('--num_classes', type=int, default=24, 
                    help='number of classes used')
parser.add_argument('--seed', type=int, default=123, 
                    help='random seed')

parser.add_argument('--model_path', type=str, default='./checkpoint/steplr_snr16_full/best_checkpoint.pth')

parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--device', type=str, default='cuda:2')

args = parser.parse_args()

### preprocess 

### train

### model

def forward(model, dataloader, criterion, epoch=0, optimizer=None, training=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()

    for i, (x, y) in enumerate(dataloader):
        inputs = x.float().to(args.device)
        target = y.long().to(args.device)[:, 0]
    
        data_time.update(time.time() - end)
        output = model(inputs)
        loss = criterion(output, target)

        prec1, prec3 = accuracy(output.detach(), target, topk=(1, 3))
        losses.update(float(loss), inputs.size(0))
        top1.update(float(prec1), inputs.size(0))
        top3.update(float(prec3), inputs.size(0))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()  

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top3.val:.3f} ({top3.avg:.3f})\t'.format(
                             epoch, i, len(dataloader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top3=top3))
    return losses.avg, top1.avg, top3.avg

def main():

    setup_logging(os.path.join('log/log.txt'))
    print(args.test_path)
    np.random.seed(args.seed)
    # for i in range(20):
    #     start = i*1024
    #     L = 1024
    #     # input = csv2input(args.test_path, start=start, len=L).to(args.device)
    #     input = IQwav2input(args.test_path, start=start, len=L).to(args.device)
    #     scale = (L / 1024)
    #     model = ResNet_S_32x32(num_classes=args.num_classes, block=BasicBlock, depth=20).to(args.device)
    #     state_dict = torch.load(args.model_path, map_location=args.device)
    #     model.load_state_dict(state_dict=state_dict)
    #     model.eval()
    #     output = model(input) / scale
    #     print(output.detach().cpu().numpy())
    #     output2class(output)
    #     output2odd(output)
    model = ResNet1d(num_classes=args.num_classes).to(args.device)
    for i in range(20):
        start = i*1024
        L = 1024
        # input = csv2input(args.test_path, start=start, len=L).to(args.device)
        # input = IQwav2input(args.test_path, start=start, len=L).to(args.device)
        input = IQwav2input1d(args.test_path, start=start, len=L).to(args.device)
        state_dict = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(state_dict=state_dict)
        model.eval()
        output = model(input)
        print(output.detach().cpu().numpy())
        output2class(output)
        output2odd(output)

if __name__ == '__main__':
    main()