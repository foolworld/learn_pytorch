import torch
from model_save import *
# 方式1-》保存方式1，加载模型  适合自己写的模型
import torchvision
from torch import nn

model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2，加载模型 适合官方提供的模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 陷阱1
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load('net_method1.pth')
print(model)