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

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='./Dataset/')

parser.add_argument('--num_classes', type=int, default=24, 
                    help='number of classes used')
parser.add_argument('--seed', type=int, default=123, 
                    help='random seed')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--save_path', type=str, default='./checkpoint/')

parser.add_argument('--samples', type=int, default=2000)
parser.add_argument('--train_ratio', type=float, default=0.8)

parser.add_argument('--eval', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--init_lr', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+' , default=[30, 60, 80])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--snr_gate', type=int, default=0)
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
    print(output[0])
    return losses.avg, top1.avg, top3.avg

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs):
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss, train_prec1, train_prec3 = forward(model, train_loader, criterion, epoch, optimizer, training = True)
        if scheduler is not None:
            scheduler.step()
        model.eval()
        val_loss, val_prec1, val_prec3 = forward(model, val_loader, criterion, epoch, training=False)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec3:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec3:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec3=train_prec3, val_prec3=val_prec3))
        if val_prec1 > best_acc:
            best_acc = val_prec1
            torch.save(model.state_dict(), args.save_path+'best_checkpoint.pth')
        
    return best_acc

def main():

    for arg in vars(args):
        print(arg, ':', getattr(args, arg))

    setup_logging(os.path.join('log/log.txt'))

    np.random.seed(args.seed)

    train_size = np.floor(args.samples * args.train_ratio)
    val_size = args.samples - train_size
    train_dataset = RFMCDataset(dataset = args.dataset, train=True, samples_per_snr=train_size, num_classes=args.num_classes, snr_gate=args.snr_gate, transform='None')
    val_dataset = RFMCDataset(dataset = args.dataset, train=False, samples_per_snr=val_size, num_classes=args.num_classes, snr_gate=args.snr_gate, transform = 'None')

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = ResNet1d(num_classes=args.num_classes).to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80], 0.02)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, scheduler, criterion, args.epochs)

if __name__ == '__main__':
    main()