from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
sys.path.append('./FeatureLearningRotNet/architectures')

from NetworkInNetwork import NetworkInNetwork
from NonLinearClassifier import Classifier as NonLinearClassifier
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy
import random
import re
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch MNIST Certification')
parser.add_argument('--models',  type=str, help='name of models directory')
parser.add_argument('--zero_seed', action='store_true', help='Use a random seed of zero (instead of the partition index)')

args = parser.parse_args()
checkpoint_dir = 'checkpoints'
radii_dir = 'radii'
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
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False, num_workers=1)
pretrainednet  = NetworkInNetwork({'num_classes': 4, 'num_stages': 4, 'num_inchannels': 1, 'use_avg_on_conv3': False})
pretrainednet.load_state_dict(torch.load('./FeatureLearningRotNet/experiments/MNIST_RotNet_NIN4blocks/model_net_epoch200')['network'])
pretrainednet = pretrainednet.to(device)
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
    net = NonLinearClassifier({'num_classes':10, 'nChannels':192, 'cls_type':'NIN_ConvBlock3'})
    net = net.to(device)
    print(modelname)
    net = net.to(device)

    checkpoint = torch.load(modelname)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    batch_offset = 0
    with torch.no_grad():
         for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = net(pretrainednet(inputs,out_feat_keys=['conv2']))
            predictions[batch_offset:inputs.size(0)+batch_offset,i,:] = out
            if firstit:
           	    labels[batch_offset:batch_offset+inputs.size(0)] = targets
            batch_offset += inputs.size(0)
    firstit = False
torch.save({'labels': labels, 'scores': predictions},'./evaluations/'+args.models+'.pth')


