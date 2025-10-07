
import torch.nn as nn

class DWSConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWSConv3d, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        # Pointwise convolution
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class L_EB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DWSConv3d(in_channels, out_channels)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)  # Use 8 groups or fewer if channels < 8
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x

class L_EB_Strided(nn.Module):
    """L_EB with configurable stride for downsampling"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Use stride in the depthwise conv for efficient downsampling
        self.conv = DWSConv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x


class L_BNB(nn.Module):
    def __init__(self, in_channels):
        super(L_BNB, self).__init__()
        mid_channels = in_channels // 4

        # Bottleneck structure with depthwise separable convolutions
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1)  # Dimension reduction
        # Changed from InstanceNorm3d to GroupNorm
        self.norm1 = nn.GroupNorm(min(8, mid_channels), mid_channels)

        self.conv2 = DWSConv3d(mid_channels, mid_channels)  # Spatial processing
        # Changed from InstanceNorm3d to GroupNorm
        self.norm2 = nn.GroupNorm(min(8, mid_channels), mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, in_channels, kernel_size=1)  # Dimension restoration
        # Changed from InstanceNorm3d to GroupNorm
        self.norm3 = nn.GroupNorm(min(8, in_channels), in_channels)

        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out = out + identity
        return self.act(out)


class L_DB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DWSConv3d(in_channels, out_channels)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x