# -*- coding: utf-8 -*-
# coding:utf-8
import matplotlib
matplotlib.use('Agg')
#%matplotlib inline
from matplotlib import pyplot as plt
import random
import torch
from IPython import display
import numpy as np

num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs)))
print(features)
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += torch.from_numpy(np.random.normal(0,0.01,size=labels.size()))
#用矢量图显示
def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    #设置图尺寸
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:,1].numpy(),labels.numpy(),1)
plt.savefig('test')

#读取数据
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) #样本读取顺序随机
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size,num_examples)])
        #index_select(i,j)中的i表示tensor的行列关系，i=0 for row and i=1 for column,j表示索引关系
        yield features.index_select(0,j),labels.index_select(0,j)

batch_size = 10
for X,y in data_iter(batch_size,features,labels):
    print(X,y)
    break

#w initialized with 0
w = torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype=torch.double)
b = torch.zeros(1,dtype=torch.double)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

#定义线性回归
def linreg(X,w,b):
    return torch.mm(X,w) + b

#定义损失函数,利用平方差
def squared_loss(y_hat,y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

#通过梯度更新参数
def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        print("before calculation,w's grad:")
        print(w.grad)
        print("before calculation,b's grad:")
        print(b.grad)
        l = (loss(net(X,w,b),y).sum()).double()
        print("after calculation,w's grad:")
        print(w.grad)
        print("after calculation,b's grad:")
        print(b.grad)
        l.backward()
        sgd([w,b],lr,batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
        train_l = loss(net(features,w,b),labels)
    print('epoch %d, loss %f' % (epoch+1,train_l.mean().item()))



