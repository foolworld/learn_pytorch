import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TwolayerNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=50, output_size=10):
        super(TwolayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为：{train_data_size}")
print(f"测试数据集的长度为：{test_data_size}")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = TwolayerNet().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.1)

epochs = 20
total_train_step = 0
total_test_step = 0

writer = SummaryWriter("./logs_mnist")

for epoch in range(epochs):
    print(f"————————第{epoch + 1}轮训练开始——————")

    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = net(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (output.argmax(1) == target).sum().item()
        total_train_step += 1

    avg_train_loss = train_loss / len(train_loader)
    avg_train_correct = train_correct / (len(train_loader)) * 100
    print(f"训练loss:{avg_train_loss:.4f},准确率：{avg_train_correct:.4f}")
    writer.add_scalar("loss", avg_train_loss, epoch)
    writer.add_scalar("accuracy", avg_train_correct, epoch)

    net.eval()
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target).item()
            test_correct += (output.argmax(1) == target).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_correct = test_correct / test_data_size * 100
    print(f"  测试 Loss: {avg_test_loss:.4f}, 准确率: {avg_test_correct:.2f}%")
    writer.add_scalar("Loss/test", avg_test_loss, epoch)
    writer.add_scalar("Accuracy/test", avg_test_correct, epoch)

    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': avg_train_correct,
            'test_acc': avg_test_correct,
        }, f'checkpoint_epoch_{epoch + 1}.pth')

torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_test_acc': avg_test_correct,
}, 'final_model.pth')

print(f"\n训练完成！最终测试准确率: {avg_test_correct:.2f}%")
writer.close()