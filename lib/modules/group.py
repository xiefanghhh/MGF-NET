import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupAtt(nn.Module):
    
    def __init__(self, in_channels,
                 groups=4, reduction_factor=4,
                 norm_layer=nn.BatchNorm2d):
        super(GroupAtt, self).__init__()
        self.in_channels = in_channels
        inter_channels = max(in_channels * groups // reduction_factor, 32)
        self.groups = groups
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(in_channels, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, groups, 1, groups=1)
        

    def forward(self, x):
        batch, channel = x.shape[:2]
        splited = torch.split(x, channel // self.groups, dim=1)
        gap = torch.cat([F.adaptive_avg_pool2d(i,1) for i in splited], dim=1)
        gmp= torch.cat([F.adaptive_max_pool2d(i,1) for i in splited], dim=1)
        gap = gap + gmp
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)

        atten = torch.split(atten, 1, dim=1)

        out = torch.cat([att * split for (att, split) in zip(atten, splited)], 1)
        return out.contiguous()
   
   
