from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy
import random
import re
from main_ce import set_loader
from networks.resnet_big import SupConResNet, LinearClassifier

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch CIFAR Certification')
parser.add_argument('--models',  type=str, help='name of models directory')
parser.add_argument('--simclr',  type=str, help='simclr model')
parser.add_argument('--zero_seed', action='store_true', help='Use a random seed of zero (instead of the partition index)')

args = parser.parse_args()
checkpoint_dir = 'checkpoints'
if not os.path.exists('./evaluations'):
    os.makedirs('./evaluations')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')


modelnames = list(map(lambda x: './checkpoints/'+args.models+'/'+x, list(filter( lambda x:x[0]!='.',os.listdir('./checkpoints/'+args.models)))))
num_classes = 10
predictions = torch.zeros(10000, len(modelnames),num_classes).cuda()
labels = torch.zeros(10000).type(torch.int).cuda()
firstit = True

train_loader, testloader = set_loader(type('',(object,),{'dataset': 'cifar10', 'data_folder' : './datasets/', 'batch_size' : 256, 'num_workers' : 1})())
pretrainednet = SupConResNet(name='resnet18')

ckpt = torch.load(args.simclr, map_location='cpu')
state_dict = ckpt['model']

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        pretrainednet.encoder = torch.nn.DataParallel(model.encoder)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    pretrainednet = pretrainednet.cuda()
pretrainednet.load_state_dict(state_dict)

pretrainednet.eval()
for i in range(len(modelnames)):
    modelname = modelnames[i]
    seed = int(re.findall(r"partition_.*\.pth",  modelname)[-1][10:-4])
    if (args.zero_seed):
        seed = 0
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net =  LinearClassifier(name='resnet18', num_classes=10)
    net = net.to(device)
    print(modelname)
    checkpoint = torch.load(modelname)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    batch_offset = 0
    with torch.no_grad():
         for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = net(pretrainednet.encoder(inputs))
            predictions[batch_offset:inputs.size(0)+batch_offset,i,:] = out
            if firstit:
           	    labels[batch_offset:batch_offset+inputs.size(0)] = targets
            batch_offset += inputs.size(0)
    firstit = False
torch.save({'labels': labels, 'scores': predictions},'./evaluations/'+args.models+'.pth')


