import torch
import torchvision
import argparse
import numpy as  np
parser = argparse.ArgumentParser(description='Order Data')
parser.add_argument('--dataset', default="mnist", type=str, help='dataset to order')
args = parser.parse_args()
channels =3
if (args.dataset == "mnist"):
    data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    channels = 1

if (args.dataset == "cifar"):
    data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

imgs, labels = zip(*data)
finalimgs = torch.stack(list(map((lambda x: torchvision.transforms.ToTensor()(x)), list(imgs))))
intmages = (finalimgs*255).int()

idxs = np.lexsort(intmages.reshape(intmages.shape[0],-1).numpy().transpose()) 

torch.save(idxs, "ordered_" +args.dataset+'.pth')