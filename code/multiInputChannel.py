import torch
from torch import nn
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def corr2d_multi_in(X,K):
    # 沿着X和K的第0维分别计算与相加
    res = d2l.corr2d(X[0, :, :],K[0, :, :])
    for i in range(1,X.shape[0]):
        res += d2l.corr2d(X[i,:, :],K[i,:,:])
    return res

# X = torch.tensor([[[0,1,2],[3,4,5],[6,7,8]],[[1,2,3],[4,5,6],[7,8,9]]])
# K = torch.tensor([[[0,1],[2,3]],[[1,2],[3,4]]])
#
# print(corr2d_multi_in(X,K))

#多个通道的输出
def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K])

#print：【3，2，2，2】
#第一个3是输出通道数通道数量，第二个2是输入通道数，第3、4个2是k_h和k_w
# K = torch.stack([K,K+1,K+2])
# print("K after unit:")
# print(K)
# print(K.shape)
# print(corr2d_multi_in_out(X,K))

#1*1卷积层被当作全连接层使用
def corr2d_multi_in_out_1x1(X,K):
    c_i,c_j,h,w = X.shape
    c_o = K.shape[0]
    print("X.shape before:")
    print(X)
    print(X.shape)
    X = X.view(c_i,h*w)
    print("X.shape after:")
    print(X)
    print(X.shape)
    print("Before,K's shape:")
    print(K)
    print(K.shape)
    K = K.view(c_o,c_i)
    print("After,K's shape:")
    print(K)
    Y = torch.mm(K,X)
    print(K.shape)
    return Y.view(c_o,h,w)

X = torch.rand(3,1,3,3)
K = torch.rand(2,3,1,1)

Y1 = corr2d_multi_in_out_1x1(X,K)
Y2 = corr2d_multi_in_out(X,K)

print((Y1 - Y2).norm().item() < 1e-6)

X_1 = torch.rand(3,1,3,3)
print("X1:")
print(X_1)
X_2 = torch.rand(3,3,3)
print("X2:")
print(X_2)











