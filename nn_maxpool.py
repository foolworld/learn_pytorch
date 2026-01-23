import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.maxpool1 = MaxPool2d(input)

    def forward(self,x):
        output  = self.maxpool1(x)
        return output


net = Net()
writer = SummaryWriter("logs_maxpool")
step =0;
for data in dataloader:
    imgs,targets =data
    writer.add_images("input",imgs,step)
    output= net(imgs)
    writer.add_images("output",output,step)
    step +=1

writer.close()

