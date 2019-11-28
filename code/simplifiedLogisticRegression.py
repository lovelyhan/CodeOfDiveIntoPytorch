# -*- coding: utf-8 -*-
import matplotlib
from torch import nn

matplotlib.use('Agg')
#%matplotlib inline
from matplotlib import pyplot as plt
import random
import torch
import torch.utils.data as Data
from IPython import display
import numpy as np

#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0,1,(num_examples,num_inputs)),dtype=torch.float)
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)

batch_size = 10
dataset = Data.TensorDataset(features,labels)
data_iter = Data.DataLoader(dataset,batch_size,shuffle=True)

for X,y in data_iter:
    print(X,y)
    break

#定义模型
class LinearNet(torch.nn.Module):
    def __init__(self,n_features):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(n_features,1)
    #forward 定义前向传播
    def forward(self,x):
        y = self.linear(x)
        return y

from collections import OrderedDict
net = nn.Sequential(
    nn.Linear(num_inputs,1)
)

#查看参数
for param in net.parameters():
    print(param)

#初始化模型参数
from torch.nn import init
init.normal_(net[0].weight,mean = 0,std = 0.01)
init.constant_(net[0].bias,val = 0)

#定义损失函数,使用均方差损失作为损失函数
loss = nn.MSELoss()

import torch.optim as optim

#利用torch提供的优化算法进行优化，提供包含SGD、ADA、RMSProp等
optimizer = optim.SGD(net.parameters(),lr=0.03)
print(optimizer)

# #动态调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1

#训练模型
num_epochs = 3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        output = net(X)
        l = loss (output,y.view(-1,1))
        #梯度清0
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d,lo    ss %f' %(epoch,l.item()))

#比较模型参数与真实参数
dense = net[0]
print(true_w,dense.weight)
print(true_b,dense.bias)









