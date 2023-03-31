import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, NUM_CLASS, init_weight):   # NUM_CLASS的值根具实际情况更改，init_weight为一个布尔值（代表是否初始化权重参数，一般为True）
        super(AlexNet, self).__init__()
        # 定义网络结构
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=2),  # 卷积层
            nn.ReLU(inplace=True),  # ReLu激活函数
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout随机丢弃神经元，0.5代表随机丢弃50%
            nn.Linear(256*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, NUM_CLASS)
        )
        if init_weight:
            self.define_weight()

    # 前向传播函数
    def forward(self, x):
        x = self.features(x)  # (3,224,224) -> (256,6,6)
        x = torch.flatten(x, start_dim=1)  # (256,6,6) -> (9216)  # 9216 = 256*6*6
        x = self.classifier(x)  # (9216) -> (10)
        return x

    def define_weight(self):  # 权重初始化参数
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                nn.init.kaiming_normal_(i.weight, mode='fan_out')
                if i.bias is not None:
                    nn.init.constant_(i.bias, 0)
            elif isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, 0, 0.01)  # 全连接层初始为正态分布均值为0方差为0.01
                nn.init.constant_(i.bias, 0)  # 用单位矩阵来填充2维输入张量或变量。在线性层尽可能多的保存输入特性。
