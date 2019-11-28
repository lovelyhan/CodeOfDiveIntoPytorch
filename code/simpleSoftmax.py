import torch
from torch import nn
from torch.nn import init
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs)
    def __forward__(self,x):
        y = self.linear(x.view(x.shape[0],-1))
        print("x.shape[0]:")
        print(x.shape[0])
        return y

net = LinearNet(num_inputs,num_outputs)

#对x进行形状转换
class FlatternLayer(nn.Module):
    def __init__(self):
        super(FlatternLayer,self).__init__()
    def forward(self,x):#x shape:(batch,*,*,...)
        return x.view(x.shape[0],-1)

#定义模型
from collections import OrderedDict
net = nn.Sequential(
    OrderedDict([
        ('flatten',FlatternLayer()),
        ('linear',nn.Linear(num_inputs,num_outputs))
    ])
)

#初始化随机权重参数
init.normal_(net.linear.weight,mean = 0,std = 0.01)
init.constant_(net.linear.bias,val = 0)

#交叉熵损失
loss = nn.CrossEntropyLoss()

#优化算法
optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)

#训练模型
num_epochs = 5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)