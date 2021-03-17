from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
sys.path.append('./FeatureLearningRotNet/architectures')

from NetworkInNetwork import NetworkInNetwork
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

args = parser.parse_args()

dirbase = 'mnist_nin_baseline'
if (args.zero_seed):
    dirbase += '_zero_seed'

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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

partitions_file = torch.load('partitions_hash_mean_mnist_'+str(args.num_partitions)+'.pth')
partitions = partitions_file['idx']
means = partitions_file['mean']
stds = partitions_file['std']

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
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
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(means[part], stds[part])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means[part], stds[part])
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    nomtestloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=1)
    print('here')
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,part_indices), batch_size=128, shuffle=True, num_workers=1)
    net  = NetworkInNetwork({'num_classes':10, 'num_inchannels': 1})
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=curr_lr, momentum=0.9, weight_decay=0.0005, nesterov= True)

# Training
    net.train()
    for epoch in range(200):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch in [60,120,160]):
            curr_lr = curr_lr * 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

    net.eval()

    (inputs, targets)  = next(iter(nomtestloader)) #Just use one test batch
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
            #breakpoint()
        outputs = net(inputs)
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
        'partition': part,
        'norm_mean' : means[part],
        'norm_std' : stds[part]
    }
    torch.save(state, checkpoint_subdir + '/partition_'+ str(part)+'.pth')




