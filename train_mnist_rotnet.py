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


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--num_partitions', default = 1200, type=int, help='number of partitions')
parser.add_argument('--start_partition', required=True, type=int, help='partition number')
parser.add_argument('--num_partition_range', default=300, type=int, help='number of partitions to train')
parser.add_argument('--zero_seed', action='store_true', help='Use a random seed of zero (instead of the partition index)')
parser.add_argument('--hash_partitions', action='store_true', help='Use hashing to partiton data')

args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dirbase = 'mnist_rotnet'
if (args.zero_seed):
    dirbase += '_zero_seed'
if (args.hash_partitions):
    dirbase += '_hash'
checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
checkpoint_subdir = f'./{checkpoint_dir}/' + dirbase + f'_partitions_{args.num_partitions}'
if not os.path.exists(checkpoint_subdir):
    os.makedirs(checkpoint_subdir)
print("==> Checkpoint directory", checkpoint_subdir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])
if (args.hash_partitions):
    partitions_file = torch.load('partitions_hash_mean_mnist_'+str(args.num_partitions)+'.pth')
else:
    partitions_file = torch.load('partitions_unsupervised_mnist_'+str(args.num_partitions)+'.pth')
partitions = partitions_file['idx']
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

nomtestloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=1)
for part in range(args.start_partition,args.start_partition+args.num_partition_range):
    seed = part
    if (args.zero_seed):
        seed = 0
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    curr_lr = 0.1
    print('\Partition: %d' % part)
    part_indices = torch.tensor(partitions[part])

    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,part_indices), batch_size=128, shuffle=True, num_workers=1)
    pretrainednet  = NetworkInNetwork({'num_classes': 4, 'num_stages': 4, 'num_inchannels': 1, 'use_avg_on_conv3': False})
    pretrainednet.load_state_dict(torch.load('./FeatureLearningRotNet/experiments/MNIST_RotNet_NIN4blocks/model_net_epoch200')['network'])
    pretrainednet = pretrainednet.to(device)
    pretrainednet.eval()
    net = NonLinearClassifier({'num_classes':10, 'nChannels':192, 'cls_type':'NIN_ConvBlock3'})
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=curr_lr, momentum=0.9, weight_decay=0.0005, nesterov= True)

# Training
    net.train()
    for epoch in range(100):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(pretrainednet(inputs,out_feat_keys=['conv2']))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch in [35,70,85]):
            curr_lr = curr_lr * 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

    net.eval()

    (inputs, targets)  = next(iter(nomtestloader)) #Just use one test batch
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
            #breakpoint()
        outputs = net(pretrainednet(inputs,out_feat_keys=['conv2']))
        loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
    acc = 100.*correct/total
    print('Accuracy: '+ str(acc)+'%') 
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'partition': part
    }
    torch.save(state, checkpoint_subdir + '/partition_'+ str(part)+'.pth')




