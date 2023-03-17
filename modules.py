def get_max(splited):
    an = []
    for i in splited:
        a1, _ = torch.max(i, dim=1, keepdim=True)
        an.append(a1)
    return torch.cat(an, dim=1)


class GroupFusion(nn.Module):
    def __init__(self, in_channels,
                 groups=4, reduction_factor=4,
                 norm_layer=nn.BatchNorm2d):
        super(GroupFusion, self).__init__()
        self.in_channels = in_channels
        inter_channels = max(in_channels * groups // reduction_factor, 32)
        self.groups = groups
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(groups, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, groups, 1, groups=1)

    def forward(self, x):
        batch, channel = x.shape[:2]
        splited = torch.split(x, channel // self.groups, dim=1)
        gap = torch.cat([torch.mean(i, dim=1, keepdim=True) for i in splited], dim=1)
        sa = get_max(splited)
        gap = gap + sa
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        atten = torch.split(atten, 1, dim=1)
        out = torch.cat([att * split for (att, split) in zip(atten, splited)], 1)
        return out.contiguous()


class EBWA(nn.Module):
    def __init__(self, in_channel, channel):
        super(EBWA, self).__init__()
        self.channel = channel

        self.conv_out1 = conv(3 * in_channel, in_channel, 3, relu=True)
        self.conv_out2 = conv(in_channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)
        self.query_conv = Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_channel + channel, out_channels=in_channel // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=3 * in_channel, out_channels=in_channel, kernel_size=1)

        self.gamma = Parameter(torch.ones(1))
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.softmax = Softmax(dim=-1)

    def forward(self, x1, x2):
        fi = self.conv(x2)
        x1 = self.ret()
        x = torch.cat([x1, x2], dim=1)
        Q = self.key_conv(x)
        K = self.query_conv(x1)
        map = F.interpolate(fi, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)
        p = fg - 0.5 ** self.gamma
        bg = 1 - p  # background
        cg = torch.abs(torch.abs(p) - 0.5 ** self.gamma)
        x1 = torch.mul(x, p)
        x2 = torch.mul(x, bg)
        x3 = torch.mul(x, cg)
        x = torch.cat([x1, x2, x3], dim=1)
        V = self.value_conv(x)

        x = self.conv_out1(x)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        return x, out + map