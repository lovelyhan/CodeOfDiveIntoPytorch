# -*- coding: utf-8 -*-
import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

#获取数据集
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

#定义模型参数
num_inputs,num_outputs,num_hiddens = 784,10,256
W1 = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_hiddens)),dtype=torch.float)
b1 = torch.zeros(num_hiddens,dtype=torch.float)
W2 = torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_outputs)),dtype=torch.float)
b2 = torch.zeros(num_outputs,dtype=torch.float)

params = [W1,b1,W2,b2]

for param in params:
    param.requires_grad_(requires_grad=True)

#定义激活函数
def relu(X):
    return torch.max(input=X,other=torch.tensor(0.0))

#定义优化函数sgd
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


#定义模型
def net(X):
    X = X.view((-1,num_inputs))
    H = relu(torch.matmul(X,W1) + b1)
    return torch.matmul(H,W2) + b2

loss = torch.nn.CrossEntropyLoss()
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

#训练模型，超参中迭代周期数为5，学习率为100
num_epochs,lr = 5,100.0
def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,
              params=None,lr=None,optimizer=None):
    print("start!")
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n = 0.0,0.0,0
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y).sum()
            # print("loss calculated:")
            # print(l)

            #梯度清0
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params,lr,batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter,net)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f'
              %(epoch + 1,train_l_sum / n,train_acc_sum / n,test_acc))


train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
