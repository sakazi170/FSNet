
import torch.nn as nn


class PatchPartition(nn.Module):
    def __init__(self, channels):
        super(PatchPartition, self).__init__()
        self.positional_encoding = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

    def forward(self, x):
        x = self.positional_encoding(x)
        return x


class LineConv(nn.Module):
    def __init__(self, channels):
        super(LineConv, self).__init__()
        expansion = 4
        self.line_conv_0 = nn.Conv3d(channels, channels * expansion, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.line_conv_1 = nn.Conv3d(channels * expansion, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.line_conv_0(x)
        x = self.act(x)
        x = self.line_conv_1(x)
        return x


class LocalRepresentationsCongregation(nn.Module):
    def __init__(self, channels):
        super(LocalRepresentationsCongregation, self).__init__()
        self.bn1 = nn.BatchNorm3d(channels)
        self.pointwise_conv_0 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv3d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.pointwise_conv_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pointwise_conv_1(x)
        return x


class usb(nn.Module):
    """Ultra Slim Block using first 3 components of Slim UNet-R"""

    def __init__(self, in_channels, out_channels):
        super(usb, self).__init__()

        # Channel adjustment if needed
        if in_channels != out_channels:
            self.channel_adjust = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_adjust = nn.Identity()

        # Ultra slim components
        self.patch1 = PatchPartition(out_channels)
        self.local_rc = LocalRepresentationsCongregation(out_channels)
        self.line_conv1 = LineConv(out_channels)

    def forward(self, x):
        # Adjust channels if needed
        x = self.channel_adjust(x)

        # Ultra slim operations with residual connections
        x = self.patch1(x) + x
        x = self.local_rc(x) + x
        x = self.line_conv1(x) + x

        return x