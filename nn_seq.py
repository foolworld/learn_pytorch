from torch import nn
from torch.nn import Conv2d,MaxPool2d
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(3, 32, 5,padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5,padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5,padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024,64)
        self.linear2 = nn.Linear(64,10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

net = Net()
print(net)
input = torch.ones((64,3,32,32))
output = net(input)
print(output.shape)