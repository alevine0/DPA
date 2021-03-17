import torch
import torchvision
import argparse
import numpy as  np
import PIL
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
if (args.dataset != "gtsrb"):
	imgs, labels = zip(*data)
	finalimgs = torch.stack(list(map((lambda x: torchvision.transforms.ToTensor()(x)), list(imgs))))
	for_sorting = (finalimgs*255).int()
	intmagessum = for_sorting.reshape(for_sorting.shape[0],-1).sum(dim=1) % args.partitions
	
else:
	labels = [label for x,label in data]
	imgs_scaled = [torchvision.transforms.ToTensor() ( torchvision.transforms.Resize((48,48),interpolation=PIL.Image.BILINEAR )(image)) for image, y in data]
	#imgs_scaled = [torchvision.transforms.ToTensor() ( torchvision.transforms.Resize((48,48),interpolation=PIL.Image.BILINEAR )(PIL.ImageOps.equalize(image))) for image, y in data] # To use histogram equalization
	finalimgs =  torch.stack(list(imgs_scaled))
	intmagessum = torch.stack([(torchvision.transforms.ToTensor()(image).reshape(-1)*255).int().sum()% args.partitions for image, y in data])
	for_sorting =finalimgs


idxgroup = list([(intmagessum  == i).nonzero() for i in range(args.partitions)])

#force index groups into an order that depends only on image content  (not indexes) so that (deterministic) training will not depend initial indices
idxgroup = list([idxgroup[i][np.lexsort(torch.cat((torch.tensor(labels)[idxgroup[i]].int(),for_sorting[idxgroup[i]].reshape(idxgroup[i].shape[0],-1)),dim=1).numpy().transpose())] for i in range(args.partitions) ])

idxgroupout = list([x.squeeze().numpy() for x in idxgroup])
means = torch.stack(list([finalimgs[idxgroup[i]].permute(2,0,1,3,4).reshape(channels,-1).mean(dim=1) for i in range(args.partitions) ]))
stds =  torch.stack(list([finalimgs[idxgroup[i]].permute(2,0,1,3,4).reshape(channels,-1).std(dim=1) for i in range(args.partitions) ]))
out = {'idx': idxgroupout,'mean':means.numpy(),'std':stds.numpy() }
torch.save(out, "partitions_hash_mean_" +args.dataset+'_'+str(args.partitions)+'.pth')