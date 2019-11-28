import torch
from torch import nn

# print("GPU Available:")
# print(torch.cuda.is_available())
# print("GPU available:")
# print(torch.cuda.device_count())
# print("current device order:")
# print(torch.cuda.current_device())
# print("current device name:")
# print(torch.cuda.get_device_name(0))

# x = torch.tensor([1,2,3])
# x = x.cuda(0)
# print(x)
# print(x.device)
# y = x**2
# print(y)

x = torch.rand(2,3).cuda()
net = nn.Linear(3,1)
net.cuda()
print(net(x))
