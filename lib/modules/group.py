import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class GroupAtt00(nn.Module):

    def __init__(self, in_channels,
                 groups=4, reduction_factor=4,
                 norm_layer=nn.BatchNorm2d):
        super(GroupAtt00, self).__init__()
        inter_channel=in_channels // groups
        self.groups = groups
        self.ca=ChannelAttention(inter_channel)
        self.sa=SpatialAttention()
    def forward(self, x):
        batch, channel = x.shape[:2]
        splited = torch.split(x, channel // self.groups, dim=1)
        gap =[self.ca(i)*i for i in splited]
        atten  =[self.sa(i)*i for i in gap]
        out = torch.cat([att * split for (att, split) in zip(atten, splited)], 1)
        return out.contiguous()

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def fmax(splited):
    an=[]
    for i in splited:
        a1,_=torch.max(i, dim=1, keepdim=True)
        an.append(a1)
    return torch.cat(an, dim=1)

class GroupAtt(nn.Module):
    """
    Split-Attend-Merge-Stack agent
    Input an feature map with shape H*W*C, we first split the feature maps into
    multiple parts, obtain the attention map of each part, and the attention map
    for the current pyramid level is constructed by mergiing each attention map.
    """

    def __init__(self, in_channels,
                 groups=4, reduction_factor=4,
                 norm_layer=nn.BatchNorm2d):
        super(GroupAtt, self).__init__()
        self.in_channels = in_channels
        inter_channels = max(in_channels * groups // reduction_factor, 32)
        self.groups = groups
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(groups, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, groups, 1, groups=1)
        # self.sa= SpatialAttention()

    def forward(self, x):
        batch, channel = x.shape[:2]
        splited = torch.split(x, channel // self.groups, dim=1)
        gap = torch.cat([torch.mean(i, dim=1, keepdim=True) for i in splited], dim=1)
        # gap = torch.cat([torch.mean(i, dim=1, keepdim=True) for i in splited], dim=1)
        sa=fmax(splited)
        gap=gap+sa
        gap= F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        # atten=atten+sa
        atten = torch.split(atten, 1, dim=1)

        out = torch.cat([att * split for (att, split) in zip(atten, splited)], 1)
        return out.contiguous()
class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
    
class GAM(nn.Module):
    def __init__(self, gate_channel,groups=4):
        super(GAM, self).__init__()
        self.channel_att = GroupAtt(gate_channel,groups=4)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + torch.sigmoid( self.channel_att(in_tensor) + self.spatial_att(in_tensor) )
        return att * in_tensor
