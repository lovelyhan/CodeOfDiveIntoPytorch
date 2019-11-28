# -*- coding: utf-8 -*-
import matplotlib
import torch
from torch import nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

class FancyMLP(nn.Module):
    def __init__(self,**kwargs):
        super(FancyMLP,self).__init__(**kwargs)
        #不可训练de参数
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self,x):
        x = self.linear(x)
        #使用创建的常数参数以及nn.functional的relu和mm,最后一维应该是常数
        x = nn.functional.relu(torch.mm(x,self.rand_weight.data)+1)
        #复用全连接层
        print("x:")
        print(x)
        x = self.linear(x)
        print("x after update:")
        print(x)
        #控制流，用item函数返回标量进行比较
        print("x.norm:")
        print(x.norm().item)
        while x.norm().item() > 1:
            x /= 2
            if x.norm().item() < 0.8:
                x *= 10
            return x.sum()

X = torch.rand(2,20)
net = FancyMLP()
print(net)
net(X)