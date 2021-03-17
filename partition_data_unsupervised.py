import torch
import torchvision
import argparse
import numpy as  np
from gtsrb_dataset import GTSRB

parser = argparse.ArgumentParser(description='Partition Data')
parser.add_argument('--dataset', default="mnist", type=str, help='dataset to partition')
parser.add_argument('--partitions', default=1200, type=int, help='number of partitions')
args = parser.parse_args()
channels =3
if (args.dataset == "mnist"):
	data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
	channels = 1

if (args.dataset == "cifar"):
    data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
if (args.dataset == "gtsrb"):
	data = GTSRB('./data', train=True)

imgs, labels = zip(*data)
if (args.dataset == "gtsrb"):
	# To save memory, we truncate images: note that this does not affect lexical sort, because (as validated) it does not result in repeats
	finalimgs = torch.stack(list(map((lambda x: torch.nn.functional.pad(torchvision.transforms.ToTensor()(x) ,(max(0,250-x.width),0,max(0,250-x.height),0) ) [:,250-16:,:]), list(imgs))))
else:
	finalimgs = torch.stack(list(map((lambda x: torchvision.transforms.ToTensor()(x)), list(imgs))))
intmages = (finalimgs*255).int()
fimages = intmages.reshape(intmages.shape[0],-1)
idxs = np.lexsort(fimages.numpy().transpose()) 
if torch.any(torch.all(fimages[idxs[1:]].eq(fimages[idxs[:intmages.shape[0]-1]]),dim=1)).item():
	raise Exception('Not implemented for repeat images for label-flipping robustness certificate.')

idxgroupout = list([idxs[i::args.partitions] for i in range(args.partitions)])
out = {'idx': idxgroupout }
torch.save(out, "partitions_unsupervised_" +args.dataset+'_'+str(args.partitions)+'.pth')