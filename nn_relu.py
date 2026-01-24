import torch
import torch.nn as nn
import torchvision
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                     [-1,3]])

input =torch.reshape(input,(-1,1,2,2))

print(input.shape)
dataset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        output = self.sigmoid(x)
        return output

net = Net()
# output = net(input)
# print(output)

writer = SummaryWriter("../logs_relu")
step =0
for data in dataloader:
    imgs,labels = data
    writer.add_images("inpu",imgs,global_step=step)
    output = net(imgs)
    writer.add_images("outpu",output,global_step=step)
    step +=1

writer.close()



