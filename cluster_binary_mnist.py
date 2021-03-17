import torch
import torchvision
import sklearn.cluster
import argparse
import numpy as np
indices = torch.load('one_seven_indices.pth')
trainset = torch.utils.data.Subset(torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=torchvision.transforms.ToTensor()), indices)
a = list(trainset)
b = list(zip(*a))
data = b[0]
data =torch.stack(data)
data  = data.reshape(13007,-1)
data = data
np.random.seed(0)
intmages = (data*255).int()
idxs = np.lexsort(intmages.reshape(intmages.shape[0],-1).numpy().transpose()) 
data = data[idxs].cpu().numpy()
b[1] = torch.tensor(b[1])[idxs].numpy()
means= sklearn.cluster.KMeans(n_clusters=2, init='random').fit(data)
fakelabels =  means.labels_
fake = torch.tensor(fakelabels)
print(str((torch.tensor(b[1]) == (fake*6+1)).sum()) + ' out of 13007')
robraw = (torch.tensor(b[1]) == (fake*6+1)).sum()
torch.save(fake, 'binary_mnist_clusters_imagespace.pth')
indices = torch.load('one_seven_indices_test.pth')

testset = torch.utils.data.Subset(torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=torchvision.transforms.ToTensor()), indices)

at = list(testset)
bt = list(zip(*at))
datat = bt[0]
lentest = len(datat)
datat =torch.stack(datat)
datat  = datat.reshape(lentest,-1)
datat = datat.cpu().numpy()

faketest = means.predict(datat)
faketest = torch.tensor(faketest)

print(str((torch.tensor(bt[1]) == (faketest*6+1)).sum()) + ' out of ' + str(lentest))
accraw = (torch.tensor(bt[1]) == (faketest*6+1)).sum()

if (robraw < 6504):
	robraw = 13007 - robraw
	accraw  = lentest - accraw

certacc = robraw- 6504

print('Robust to ' + str(certacc) + ' out of 13007 label flips. (' + str(certacc*100./13007)+'%)')
print('Accuracy is'  + str(accraw) + ' out of ' +str(lentest)+'. (' + str(accraw*100./lentest)+'%)')



