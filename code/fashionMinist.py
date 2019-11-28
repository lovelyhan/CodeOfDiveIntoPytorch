# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
#%matplotlib inline
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms
import time
import sys
#导入d2lzh_pytorch
sys.path.append("..")
import d2lzh_pytorch as d2l


mnist_train = torchvision.datasets.FashionMNIST(root = '~/Datasets/FashionMNIST',train = True,download=True,transform = transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=False,download=True,transform=transforms.ToTensor())

#打印训练集和测试集个数
print(type(mnist_train))
print(len(mnist_train),len(mnist_test))

#获取图像的类别信息
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankleboot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images,labels):
    d2l.use_svg_display()
    #用_表示忽略变量
    _,figs = plt.subplots(1,len(images),figsize = (12,12))
    for f,img,lbl,in zip(figs,images,labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.savefig('fashionMinist')

X,y = [],[]
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X,get_fashion_mnist_labels(y))

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)

start = time.time()
for X,y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))






