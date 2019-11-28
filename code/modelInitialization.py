import torch
from torch import nn
from torch.nn import init
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# net = nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,1))

#访问所有模型参数，以迭代器形式返回
# print("type(net.named_parameters()):")
# print(type(net.named_parameters()))
# for name,param in net.named_parameters():
#     print("name,param.size:")
#     print(name,param.size())

# #权重参数初始化成均值=0、标准差=0.01的正态分布随机数
# for name,param in net.named_parameters():
#     if 'weight' in name:
#         init.normal_(param,mean = 0,std = 0.01)
#         print(name,param.data)
#
# #使用常数初始化权重参数
# for name,param in net.named_parameters():
#     if 'bias' in name:
#         init.constant_(param,val = 0)
#         print(name,param.data)

# #自定义初始化参数
# def init_weight_(tensor):
#     with torch.no_grad():
#         tensor.uniform_(-10,10)
#         tensor *= (tensor.abs() >= 5).float()
#
# for name,param in net.named_parameters():
#     if 'weight' in name:
#         init_weight_(param)
#         print(name,param.data)

#共享模型参数
linear = nn.Linear(1,1,bias=False)
net = nn.Sequential(linear,linear)
print(net)
for name ,param in net.named_parameters():
    init.constant_(param,val = 3)
    print(name,param.data)

print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))

x = torch.ones(1,1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)







