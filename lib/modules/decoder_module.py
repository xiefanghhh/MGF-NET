import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *
from .group import *


class PAA_d0(nn.Module):
    # dense decoder, it can be replaced by other decoder previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(PAA_d0, self).__init__()
        self.conv1 = conv(channel * 3, channel, 3)
        self.conv2 = conv(channel, channel, 3)
        self.conv3 = conv(channel, channel, 3)
        self.conv4 = conv(channel, channel, 3)
        self.conv5 = conv(channel, 1, 3, bn=False)

        self.se = GroupAtt(channel * 3)

        # self.Hattn = self_attn(channel, mode='h')
        # self.Wattn = self_attn(channel, mode='w')

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)

    def forward(self, f3, f2, f1):
        f3 = self.upsample(f3, f1.shape[-2:])
        f2 = self.upsample(f2, f1.shape[-2:])
        f3X2 = torch.mul(f3,f2)
        f3X1 = torch.mul(f3X2,f1)
        f3 = torch.cat([f3,f3X2 ,f3X1], dim=1)
        # f3 = self.se(f3)
        f3 = self.conv1(f3)
        f3 = self.conv2(f3)
        f3 = self.conv3(f3)
        f3 = self.conv4(f3)
        out = self.conv5(f3)

        return f3, out