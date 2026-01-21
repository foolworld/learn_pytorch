import torch
from torch import nn


class Moudle(nn.Module):
    def __init__(self):
        super(Moudle, self).__init__()

    def forward(self, input):
        output = input +1
        return output

nn = Moudle()
x=torch.tensor(1.0)
output=nn(x)
print(output)
