import torch
from torch import nn
from torch.nn import init

#定义含模型参数的自定义层
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense,self).__init__()
        self.params = nn.ParameterDict({
            'linear1' : nn.Parameter(torch.randn(4,4)),
            'linear2' : nn.Parameter(torch.randn(4,1))
        })
        #通过update新增参数
        self.params.update({'linear3':nn.Parameter(torch.randn(4,2))})
        # print("self.param:")
        # print(self.params)
        # print("nn.Parameter(torch.randn(4,1)")
        # #利用append可以增加paramlist的参数
        # self.params.append(nn.Parameter(torch.randn(4,1)))
        # print(nn.Parameter(torch.randn(4,1)))

    def forward(self,x, choice='linear1'):
        return torch.mm(x,self.params[choice])

net = MyDense()
print(net)

x = torch.ones(1,4)
print(net(x,'linear1'))
print(net(x,'linear2'))
print(net(x,'linear3'))
