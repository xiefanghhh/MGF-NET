import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax
from lib.modules.layers import *
from utils.utils import *

class EBWA(nn.Module):
    def __init__(self, in_channel, channel):
        super(EBWA, self).__init__()
        self.channel = channel

        self.conv_out1 = conv(3*in_channel,in_channel, 3, relu=True)
        self.conv_out2 = conv(in_channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)
        self.query_conv = conv(in_channel, in_channel // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channel, in_channel // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channel, in_channel, kernel_size=(1, 1))
        self.gamma = Parameter(torch.ones(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重
       
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)
        p = fg -  0.5**self.gamma
        bg = 1-p # background
        cg = torch.abs(torch.abs(p) -0.5)# confusion area
       
        
        
        x1=torch.mul(x,p)
        x2=torch.mul(x,bg)
        x3=torch.mul(x,cg)
        
        x=torch.cat([x1,x2,x3],dim=1)
        
        x=self.conv_out1(x)
        x=self.conv_out2(x)
        x=self.conv_out3(x)
        out=self.conv_out4(x)
        return x,out+map
