import torch.nn as nn

class simplecnn(nn.Module):
    def __init__(self, num_class): # num_class: number of classes
        super().__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # 保持图像大小不变 16*224*224
            nn.ReLU(), # 卷积层后面加一个激活函数，增加非线性
            nn.MaxPool2d(kernel_size=2, stride=2), # 池化层，减小图像大小 16*112*112
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 保持图像大小不变 32*112*112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 池化层，减小图像大小 32*56*56
        )

        # 全连接层（分类层）
        self.classifier = nn.Sequential(
            nn.Linear(32*56*56, 128), # 32*56*56 -> 128
            nn.ReLU(),
            nn.Linear(128, num_class) # 128 -> num_class
        )

        # 前向传播
    def forward(self, x):
        x = self.features(x) # 特征提取层
        x = x.view(x.size(0), -1) # 将特征图展平, x.size(0)是batch_size, -1表示（32*56*56）
        x = self.classifier(x) # 分类层
        return x    
