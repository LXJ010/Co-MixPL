import torch
import torch.nn as nn
from collections import OrderedDict


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv3x3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.ln = nn.LayerNorm(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv3x3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        return x


class calcFPN(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.downsample1 = Downsample(in_channels=in_channels)
        self.upsample1 = Upsample(in_channels=in_channels)

    def forward(self, x, featurealign=False):
        if featurealign:
            u1 = self.upsample1(x[0])
            x.insert(0, u1)
            names = ['0', '1', '2', '3', 'pool']
            return OrderedDict([(k, v) for k, v in zip(names, x)])
        d1 = self.downsample1(x[-1])
        x.append(d1)
        names = ['0', '1', '2', '3', 'pool']
        return OrderedDict([(k, v) for k, v in zip(names, x)])


def get_test_fpn():
    batch, in_channels, h, w = 4, 256, 32, 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_map = torch.randn(batch, in_channels, h, w).to(device)
    downsample_model = calcFPN(in_channels=in_channels)
    downsample_model = downsample_model.to(device)
    return downsample_model(feature_map)


if __name__ == '__main__':
    a = get_test_fpn()
    print(a)