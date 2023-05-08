import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .optim.losses import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax
from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s

class MGFNet(nn.Module):
    def __init__(self, channels=256, output_stride=16, pretrained=True):
        super(MGFNet, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained, output_stride=output_stride)
        self.context2 = RFB_GAP(512, channels,2)
        self.context3 = RFB_GAP(1024, channels,4)
        self.context4 = RFB_GAP(2048, channels,8)

        self.decoder = PAA_d0(channels)

        self.attention2 = UACA0(channels * 2, channels)
        self.attention3 = UACA0(channels * 2, channels)
        self.attention4 = UACA0(channels * 2, channels)

        self.loss_fn = bce_iou_loss

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):

        
        base_size = x.shape[-2:]
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)

        f5, a5 = self.decoder(x4, x3, x2)
        out5 = self.res(a5, base_size)

        f4, a4 = self.attention4(torch.cat([x4, self.ret(f5, x4)], dim=1), a5)
        out4 = self.res(a4, base_size)

        f3, a3 = self.attention3(torch.cat([x3, self.ret(f4, x3)], dim=1), a4)
        out3 = self.res(a3, base_size)

        f2, a2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), a3)
        out2 = self.res(a2, base_size)
       
        
        return out2,out3,out4,out5

        