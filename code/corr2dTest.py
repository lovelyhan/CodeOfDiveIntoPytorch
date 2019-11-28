import torch
from torch import nn

def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1,X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i + h,j:j + w]* K).sum()
    return Y

#定义二维卷积层
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D,self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self,x):
        return corr2d(x,self.weight) + self.bias

X = torch.ones(6,8)
K = torch.tensor([[1,-1]])
X[:, 2:6] = 0
# print(X)
#
# #创建高、宽分别为1、2的卷积核K
# #若横向相邻元素相同，输出0否则输出非0
# K = torch.tensor([[1,-1]])
Y = corr2d(X,K)
print("Y:")
print(Y)

#构造一个核数组形状是（1，2）的二维卷积层
#在卷积层中初始化超参w与b
conv2d = Conv2D(kernel_size=(1,2))

step = 20
lr = 0.01
for i in range(step):
    print("X:")
    print(X)
    Y_hat = conv2d(X)
    print("conv2d(X):")
    print(conv2d(X))
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    #梯度下降
    conv2d.weight.data -= lr*conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
    #梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if(i+1) % 5 == 0:
        print('Step %d ,loss %.3f' %(i + 1,l.item()))
