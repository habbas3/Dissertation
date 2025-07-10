
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiHeadAttention as attention
from models.Self_Attention import SelfAttention as attention_sa
from models.Self_Attention import OutlierAttention as attention_outliers


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

class WideResNet_sa(nn.Module):
    def __init__(self, depth, widen_factor, drop_rate, num_classes, num_input_channels=7):
        super(WideResNet_sa, self).__init__()
        
        self.nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        self.conv1 = nn.Conv1d(num_input_channels, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, self.nChannels[0], self.nChannels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, self.nChannels[1], self.nChannels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, self.nChannels[2], self.nChannels[3], block, 2, drop_rate)

        self.bn1 = nn.BatchNorm1d(self.nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.attn = attention_sa(self.nChannels[3])
        self.attn_outliers = attention_outliers(self.nChannels[3])
        self.fc = nn.utils.parametrizations.spectral_norm(nn.Linear(self.nChannels[3], num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = self.block1(out)
    #     out = self.block2(out)
    #     out = self.block3(out)
    #     out = self.attn(out)
    #     out = self.attn_outliers(out)
    #     out = self.relu(self.bn1(out))
    #     out = F.avg_pool1d(out, out.size(2))
    #     out = out.view(out.size(0), -1)
    #     return self.fc(out)

    # def output_num(self):
    #     return self.fc.out_features



    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        # Check for NaN values before attention layers and replace them
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            out[nan_mask] = 0  # Or another value that makes sense in your context

        # Apply attention layers
        out = self.attn(out)
        out = self.attn_outliers(out)

        # Check for NaN values after attention layers and replace them
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            out[nan_mask] = 0  # Or another value that makes sense in your context

        out = self.relu(self.bn1(out))
        out = F.avg_pool1d(out, out.size(2))
        out = out.view(out.size(0), -1)
        return self.fc(out)

    def output_num(self):
        return self.fc.out_features