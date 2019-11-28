import time
import torch
from torch import nn,optim
import torch.nn.functional as F

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define nin block
def nin_block(in_channels,out_channels,kernel_size,stride,padding):
    blk = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels,out_channels,kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels,out_channels,kernel_size=1),
                        nn.ReLU())
    return blk


class GlobalAvgPool2d(nn.Module):
    #全局平均池化层可通过池化窗口形状设置为输入的高和宽
    def __init__(self):
        super(GlobalAvgPool2d,self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

net = nn.Sequential(
    nin_block(1,96,kernel_size=11,stride=4,padding=0),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(96,256,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(256,384,kernel_size=3,stride=1,padding=1),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Dropout(0.5),
    #标签类数10
    nin_block(384,10,kernel_size=3,stride=1,padding=1),
    GlobalAvgPool2d(),
    #四维转二维
    d2l.FlattenLayer())

X = torch.rand(1,1,224,224)
for name,blk in net.named_children():
    X = blk(X)
    print(name,'output shape:',X.shape)

#获取数据、训练模型
batch_size = 128
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize = 224)

lr,num_epochs = 0.002,5
optimizer = torch.optim.Adam(net.parameters(),lr = lr)
d2l.train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)
