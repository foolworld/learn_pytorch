import torch
from torch import nn
from torch.nn import functional as F


# ==================== 残差块 ====================
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()

        # 第一个3x3卷积
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # 第二个3x3卷积
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

        # 1x1卷积快捷连接（需要时使用）
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        # 主路径
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        # 快捷连接
        if self.conv3:
            X = self.conv3(X)

        # 残差连接
        Y += X
        return self.relu(Y)


# ==================== ResNet-18 ====================
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # 初始层
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4个阶段，每个阶段2个残差块
        # 阶段1: 64→64, 不降采样
        self.b2 = nn.Sequential(
            Residual(64, 64, use_1x1conv=False, strides=1),
            Residual(64, 64, use_1x1conv=False, strides=1)
        )

        # 阶段2: 64→128, 第一个块降采样
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1x1conv=True, strides=2),
            Residual(128, 128, use_1x1conv=False, strides=1)
        )

        # 阶段3: 128→256, 第一个块降采样
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1x1conv=True, strides=2),
            Residual(256, 256, use_1x1conv=False, strides=1)
        )

        # 阶段4: 256→512, 第一个块降采样
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1x1conv=True, strides=2),
            Residual(512, 512, use_1x1conv=False, strides=1)
        )

        # 分类头
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        X = self.b6(X)
        return X