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

parser = argparse.ArgumentParser(description='Certification')
parser.add_argument('--evaluations',  type=str, help='name of evaluations directory')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')

args = parser.parse_args()
if not os.path.exists('./radii'):
    os.makedirs('./radii')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

filein = torch.load('./evaluations/'+args.evaluations)
labels = filein['labels']
scores = filein['scores']
num_classes = args.num_classes
max_classes = scores.max(2).indices
predictions = torch.zeros(max_classes.shape[0],num_classes)
for i in range(max_classes.shape[1]):
	predictions[(torch.arange(max_classes.shape[0]),max_classes[:,i])] += 1
predinctionsnp = predictions.cpu().numpy()
idxsort = numpy.argsort(-predinctionsnp,axis=1,kind='stable')
valsort = -numpy.sort(-predinctionsnp,axis=1,kind='stable')
val =  valsort[:,0]
idx = idxsort[:,0]
valsecond =  valsort[:,1]
idxsecond =  idxsort[:,1] 
diffs = ((val - valsecond - (idxsecond <= idx))/2).astype(int)
certs = torch.tensor(diffs).cuda()
torchidx = torch.tensor(idx).cuda()
certs[torchidx != labels] = -1
base_acc = 100 *  (max_classes == labels.unsqueeze(1)).sum().item() / (max_classes.shape[0] * max_classes.shape[1])
print('Base classifier accuracy: ' + str(base_acc))
torch.save(certs,'./radii/'+args.evaluations)
a = certs.cpu().sort()[0].numpy()
accs = numpy.array([(i <= a).sum() for i in numpy.arange(numpy.amax(a)+1)])/predictions.shape[0]
print('Smoothed classifier accuracy: ' + str(accs[0] * 100.) + '%')
print('Robustness certificate: ' + str(sum(accs >= .5)))


