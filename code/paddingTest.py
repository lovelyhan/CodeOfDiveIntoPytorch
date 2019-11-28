import torch
import torch.nn as nn

def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1,X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y

def comp_conv2d(conv2d,X):
    X = X.view((1,1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])

conv2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1)

X = torch.rand(8,8)

print(comp_conv2d(conv2d,X).shape)

