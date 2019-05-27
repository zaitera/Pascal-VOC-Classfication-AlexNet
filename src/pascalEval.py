# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: bbrattol
"""
import os, sys, numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn.metrics import average_precision_score

import tensorflow
from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
# #import torchnet as tnt
# from torchviz import make_dot
import multiprocessing
CORES = 4#int(float(multiprocessing.cpu_count())*0.25)

from PascalLoader  import DataLoader
from PascalNetwork import Network

#sys.path.append('/export/home/bbrattol/git/JigsawPuzzlePytorch/Architecture')
#from alexnet import AlexNet as Network

from TrainingUtils import adjust_learning_rate

parser = argparse.ArgumentParser(description='Train network on Pascal VOC 2007')
parser.add_argument('pascal_path', type=str, help='Path to Pascal VOC 2007 folder')
parser.add_argument('--model', default=None, type=str, help='Pretrained model')
#parser.add_argument('--freeze', dest='evaluate', action='store_true', help='freeze layers up to conv5')
parser.add_argument('--freeze', default=None, type=int, help='freeze layers up to conv5')
parser.add_argument('--fc', default=None, type=int, help='load fc6 and fc7 from model')
parser.add_argument('--gpu', default=None, type=int, help='gpu id')
parser.add_argument('--epochs', default=160, type=int, help='gpu id')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=10, type=int, help='batch size')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--crops', default=10, type=int, help='number of random crops during testing')
args = parser.parse_args()
#args = parser.parse_args([
#    '/net/hci-storage02/groupfolders/compvis/datasets/VOC2007/',
#    '--gpu','0',
#])

def compute_mAP(labels,outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    print("HERE !!! ", y_true)
    print("HERE 123!!! ", y_pred*-1)
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i],y_pred[i]))
    return np.mean(AP)

def main():
    if args.gpu is not None:
        print('Using GPU %d'%args.gpu)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    else:
        print('CPU mode')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std= [0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
            transforms.RandomSizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    val_transform = transforms.Compose([
            #transforms.Scale(256),
            #transforms.CenterCrop(227),
            transforms.RandomSizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    # DataLoader initialize
    val_data   = DataLoader(args.pascal_path,'test',transform=val_transform,random_crops=args.crops)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=args.batch, 
                                             shuffle=False,
                                             num_workers=CORES)
    
    # N = len(train_data.names)
    # iter_per_epoch = int(N/args.batch)
    # Network initialize
    #net = Network(groups = 2)
    net = Network(num_classes = 21)
    
    if args.gpu is not None:
        net.cuda()
    
    if args.model is not None:
        net.load(args.model,args.fc)
    
    if args.freeze is not None:
        # Freeze layers up to conv4
        for i, (name,param) in enumerate(net.named_parameters()):
            if 'conv' in name or 'features' in name:
                param.requires_grad = False
    
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=args.lr,momentum=0.9,weight_decay = 0.0001)
    
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint+'/train')
        os.makedirs(args.checkpoint+'/test')
    
#    logger_test = None
    logger_train = Logger(args.checkpoint+'/train')
    logger_test  = Logger(args.checkpoint+'/test')
    
    ############## validation ###############
    print('Start training: lr %f, batch size %d'%(args.lr,args.batch))
    print('Checkpoint: '+args.checkpoint)
    model_show = True
    weights = torch.load('./checkpoints/jps_156.pth')
    net.load_state_dict(weights)
    print("okay")
    test(net,criterion,logger_test,val_loader)


def test(net,criterion,logger,val_loader):
    mAP = []
    net.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = images.view((-1,3,227,227))
        images = Variable(images, volatile=True)
        if args.gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)
        outputs = outputs.cpu().data
        outputs = outputs.view((-1,args.crops,21))
        outputs = outputs.mean(dim=1).view((-1,21))
        
        #score = tnt.meter.mAPMeter(outputs, labels)
        mAP.append(compute_mAP(labels,outputs))
    
    if logger is not None:
        logger.scalar_summary('mAP', np.mean(mAP))
    print ('TESTING: mAP {}'.format(100*np.mean(mAP)))
    net.train()


if __name__ == "__main__":
    main()

