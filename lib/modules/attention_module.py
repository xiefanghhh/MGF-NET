import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax
from lib.modules.layers import *
from utils.utils import *

class reverse_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3):
        super(reverse_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 3 if kernel_size == 3 else 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        rmap = -1 * (torch.sigmoid(map)) + 1
        
        x = rmap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out

class simple_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3):
        super(simple_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        amap = torch.sigmoid(map)
        
        x = amap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out

class CA(nn.Module):
    def __init__(self, in_channel, channel):
        super(CA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x, map):
        b, c, h, w = x.shape
        
        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)
        
        p = fg - .5

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background

        prob = torch.cat([fg, bg], dim=1)

        # reshape feature & prob
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 2, h * w)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 2, c

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        out = out + map
        return x, out

class UACA(nn.Module):
    def __init__(self, in_channel, channel):
        super(UACA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x, map):
        b, c, h, w = x.shape
        
        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)
        
        p = fg - .5

        fg = torch.clip(p, 0, 1) # foreground
        bg = torch.clip(-p, 0, 1) # background
        cg = .5 - torch.abs(p) # confusion area

        prob = torch.cat([fg, bg, cg], dim=1)

        # reshape feature & prob
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 3, h * w)
        
        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3) # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key) # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        out = out + map
        return x, out
    
    
class UACA0(nn.Module):
    def __init__(self, in_channel, channel):
        super(UACA0, self).__init__()
        self.channel = channel

        self.conv_out1 = conv(3*in_channel,in_channel, 3, relu=True)
        self.conv_out2 = conv(in_channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)
      
        self.gamma = Parameter(torch.ones(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重
        # self.f1 = Parameter(torch.ones(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重
        # self.b1 = Parameter(torch.ones(1))
        
        # self.atten1=Axial_Layer(in_channel)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x, map):
        # y=torch.randn(32,1,352,252)
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)
        p = fg -  0.5**self.gamma
        bg = 1-p # background
        cg = torch.abs(torch.abs(p) -0.5)# confusion area
        
        # save_image( prob,"./picture/ prob.png" , padding =0,normalize=True)
        # sys.exit(0)
        x1=torch.mul(x,p)
        x2=torch.mul(x,bg)
        x3=torch.mul(x,cg)
        x=torch.cat([x1,x2,x3],dim=1)
        x=self.conv_out1(x)
        x=self.conv_out2(x)
        x=self.conv_out3(x)
        out=self.conv_out4(x)
        return x,out+map
