
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiHeadAttention as attention
from models.Self_Attention import SelfAttention as attention_sa

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
        #                        padding=1, bias=False)
        self.conv2 = nn.Conv1d(out_planes,out_planes,kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        # self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        #                        padding=0, bias=False) or None
        self.convShortcut = (not self.equalInOut) and nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropRate, out_features, in_channels=7, bottleneck=256):
        super(WideResNet, self).__init__()

        self.nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        self.conv1 = nn.Conv1d(in_channels, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = NetworkBlock(n, self.nChannels[0], self.nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, self.nChannels[1], self.nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, self.nChannels[2], self.nChannels[3], block, 2, dropRate)

        self.bn1 = nn.BatchNorm1d(self.nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Additional linear layer to match CNN's bottleneck feature size
        self.feature_reduction = nn.Sequential(
            nn.Linear(self.nChannels[3], bottleneck),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(bottleneck, out_features)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        features = self.feature_reduction(x)  # Ensures consistent feature size of 256
        return features

    def forward(self, x):
        features = self.forward_features(x)
        out = self.fc(features)
        return out

    def output_num(self):
        return self.fc.out_features
