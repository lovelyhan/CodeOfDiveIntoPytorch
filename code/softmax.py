# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

#使用fashion-mnist数据集，并将批量大小设置为256
batch_size = 512
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)


#输入样本是高*宽=28*28=784的图像
#图像有10个类别，softmax对应de权重和偏差参数是784*10和1*10的矩阵
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float)
b = torch.zeros(num_outputs,dtype=torch.float)

#设置模型参数梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def softmax(X):
    X_exp = X.exp()
    #按行求和dim=1
    partition = X_exp.sum(dim=1,keepdim=True)
    #广播
    return X_exp / partition

# X = torch.rand((2,5))
# X_prob = softmax(X)
# print(X_prob,X_prob.sum(dim = 1))

def net(X):
    #利用view将X展开为一行
    #参数-1表示不想自己计算的参数，由计算机代替计算完成
    return softmax(torch.mm(X.view((-1,num_inputs)),W)+b)

#
# y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
# #longTensor指明需要的index位置
# y = torch.LongTensor([0,2])
# #gather第一位指按行取，view表示将y转置为列向量
# print(y_hat.gather(1,y.view(-1,1)))

#交叉熵函数计算损失
def cross_entropy(y_hat,y):
    return - torch.log(y_hat.gather(1,y.view(-1,1)))

# print(accuracy(y_hat,y))

#判断模型准确率
#data_iter--data,net--model
def evaluate_accuracy(data_iter,net):
    acc_sum,n = 0.0,0
    for X,y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        #item:return the first element of self
        n += y.shape[0]
    return acc_sum / n

print("evaluate_accuracy:")
print(evaluate_accuracy(test_iter,net))

#训练模型
num_epochs,lr = 5,0.1

def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,
              params=None,lr=None,optimizer=None):
    print("start!")
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n = 0.0,0.0,0
        for X,y in train_iter:
            y_hat = net(X)
            l = cross_entropy(y_hat,y).sum()

            #梯度清0
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params,lr,batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter,net)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f'
              %(epoch + 1,train_l_sum / n,train_acc_sum / n,test_acc))


train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,batch_size,[W,b],lr)

#预测模型
X,y = iter(test_iter).next()
print("X:")
print(X)
print("Y:")
print(y)

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
print("True labels:")
print(true_labels)
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim = 1).numpy())
print("predic labels:")
print(pred_labels)
titles = [true + '\n' + pred for true,pred in zip(true_labels,pred_labels)]

_, figs = plt.subplots(1, len(X[0:9]), figsize=(12, 12))
for f, img, lbl in zip(figs, X[0:9], titles[0:9]):
    f.imshow(img.view((28, 28)).numpy())
    f.set_title(lbl)
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)
plt.savefig('fashionMinistResult')
