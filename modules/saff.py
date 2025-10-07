import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECA(nn.Module):
    """Efficient Channel Attention (ECA) from ECA-Net paper"""

    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # Adaptive kernel size calculation from ECA paper
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Original CBAM from paper (Channel + Spatial Attention)"""

    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Channel attention
        reduced_channels = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv3d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, channels, 1, bias=False)
        )

        # Spatial attention
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                                      padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        x = x * channel_attention

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_attention

        return x


class SpatialAttention(nn.Module):
    """Spatial Attention for local branch"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.conv(spatial_input))
        return x * spatial_attention


class DWSConv3d(nn.Module):
    """Depthwise Separable 3D Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWSConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LocalSpatialBranch(nn.Module):
    """Local-Spatial Branch: Direct DWSConv3d + Spatial Attention (NO residual)"""

    def __init__(self, in_channels, out_channels, num_blocks=2):
        super(LocalSpatialBranch, self).__init__()

        # Direct DWSConv3d blocks (no residual connection)
        self.dws_convs = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                self.dws_convs.append(DWSConv3d(in_channels, out_channels))
            else:
                self.dws_convs.append(DWSConv3d(out_channels, out_channels))

        # Spatial attention
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Sequential DWSConv3d processing (no residual)
        for dws_conv in self.dws_convs:
            x = dws_conv(x)

        # Apply spatial attention
        x = self.spatial_attention(x)
        return x


class GlobalBranch(nn.Module):
    """Global Branch with ECA inside transformer"""

    def __init__(self, in_channels, embed_dim, out_channels, num_blocks=2, patch_size=2):
        super(GlobalBranch, self).__init__()

        # Patch embedding
        self.patch_embed = nn.Conv3d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size, bias=False)

        # Transformer blocks with ECA
        self.former_blocks = nn.ModuleList([
            self._create_former_block(embed_dim) for _ in range(num_blocks)
        ])

        # Unpatchify
        self.unpatchify = nn.ConvTranspose3d(embed_dim, out_channels,
                                             kernel_size=patch_size, stride=patch_size, bias=False)

    def _create_former_block(self, channels, mlp_ratio=4):
        """Create transformer block with ECA attention"""
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(channels),
            'eca_attention': ECA(channels),  # ECA instead of standard CA
            'conv1x1': nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            'norm2': nn.LayerNorm(channels),
            'mlp': nn.Sequential(
                nn.Linear(channels, int(channels * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(channels * mlp_ratio), channels)
            )
        })

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Transformer blocks
        for block in self.former_blocks:
            identity = x

            # First part: x + ECA_Tokenizer(Norm(x))
            norm1_x = x.permute(0, 2, 3, 4, 1)
            norm1_x = block['norm1'](norm1_x).permute(0, 4, 1, 2, 3)

            eca_out = block['eca_attention'](norm1_x)
            conv_out = block['conv1x1'](eca_out)
            fo = identity + conv_out

            # Second part: fo + MLP(Norm(fo))
            norm2_fo = fo.permute(0, 2, 3, 4, 1)
            norm2_fo = block['norm2'](norm2_fo)
            mlp_out = block['mlp'](norm2_fo).permute(0, 4, 1, 2, 3)

            x = fo + mlp_out

        # Unpatchify
        x = self.unpatchify(x)
        return x


class MultiScaleFeatureStandardizer(nn.Module):
    """Standardize multi-scale encoder features to same size"""

    def __init__(self, encoder_channels, target_channels, target_size):
        super(MultiScaleFeatureStandardizer, self).__init__()
        self.target_size = target_size

        self.standardizers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, target_channels, 1, bias=False),
                nn.GroupNorm(min(8, target_channels), target_channels),
                nn.ReLU(inplace=True)
            ) for ch in encoder_channels
        ])

    def forward(self, encoder_features):
        standardized = []
        for features, standardizer in zip(encoder_features, self.standardizers):
            std_features = standardizer(features)

            # Interpolate to target size if needed
            if std_features.shape[2:] != self.target_size:
                std_features = F.interpolate(std_features, size=self.target_size,
                                             mode='trilinear', align_corners=False)
            standardized.append(std_features)

        return standardized


class SAFF(nn.Module):
    """
    Scale-Aware Feature Fusion (SAFF) Module

    Architecture:
    1. Interpolate all encoder features to bottleneck size
    2. Concatenate multi-scale features
    3. CBAM (original from paper)
    4. Parallel processing:
       - Local-Spatial Branch (DWSConv + Spatial Attention)
       - Global Branch (Transformer with ECA inside)
    5. Addition fusion
    """

    def __init__(self, encoder_channels=[16, 32, 48, 64], embed_dim=64, target_size=(8, 8, 8)):
        super(SAFF, self).__init__()

        # Step 1: Standardize multi-scale features to same size
        self.feature_standardizer = MultiScaleFeatureStandardizer(
            encoder_channels, embed_dim, target_size
        )

        concat_channels = embed_dim * len(encoder_channels)

        # Step 2: CBAM after concatenation
        self.cbam = CBAM(concat_channels)

        # Step 3: Two parallel branches
        # Local-Spatial Branch (no residual)
        self.local_spatial_branch = LocalSpatialBranch(concat_channels, embed_dim, num_blocks=2)

        # Global Branch (with ECA inside transformer)
        self.global_branch = GlobalBranch(concat_channels, embed_dim, embed_dim,
                                          num_blocks=2, patch_size=2)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of [fused_features, e1, e2, e3]
        Returns:
            Fused bottleneck features
        """
        # Step 1: Interpolate and standardize to bottleneck size
        standardized_features = self.feature_standardizer(encoder_features)

        # Step 2: Concatenate multi-scale features
        concatenated = torch.cat(standardized_features, dim=1)

        # Step 3: CBAM (channel + spatial attention)
        attended = self.cbam(concatenated)

        # Step 4: Parallel processing
        local_spatial_features = self.local_spatial_branch(attended)
        global_features = self.global_branch(attended)

        # Step 5: Addition fusion (final output)
        output = local_spatial_features + global_features

        return output

class SAFF11(nn.Module):
    """
    Scale-Aware Feature Fusion (SAFF) Module

    Architecture:
    1. Interpolate all encoder features to bottleneck size
    2. Concatenate multi-scale features
    3. CBAM (original from paper)
    4. Parallel processing:
       - Local-Spatial Branch (DWSConv + Spatial Attention)
       - Global Branch (Transformer with ECA inside)
    5. Addition fusion
    """

    def __init__(self, encoder_channels=[16, 32, 48, 64], embed_dim=64, target_size=(8, 8, 8)):
        super(SAFF11, self).__init__()

        # Step 1: Standardize multi-scale features to same size
        self.feature_standardizer = MultiScaleFeatureStandardizer(
            encoder_channels, embed_dim, target_size
        )

        concat_channels = embed_dim * len(encoder_channels)

        # Step 2: CBAM after concatenation
        self.cbam = CBAM(concat_channels)

        # Step 3: Two parallel branches
        # Local-Spatial Branch (no residual)
        self.local_spatial_branch = LocalSpatialBranch(concat_channels, embed_dim, num_blocks=1)

        # Global Branch (with ECA inside transformer)
        self.global_branch = GlobalBranch(concat_channels, embed_dim, embed_dim,
                                          num_blocks=1, patch_size=2)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of [fused_features, e1, e2, e3]
        Returns:
            Fused bottleneck features
        """
        # Step 1: Interpolate and standardize to bottleneck size
        standardized_features = self.feature_standardizer(encoder_features)

        # Step 2: Concatenate multi-scale features
        concatenated = torch.cat(standardized_features, dim=1)

        # Step 3: CBAM (channel + spatial attention)
        attended = self.cbam(concatenated)

        # Step 4: Parallel processing
        local_spatial_features = self.local_spatial_branch(attended)
        global_features = self.global_branch(attended)

        # Step 5: Addition fusion (final output)
        output = local_spatial_features + global_features

        return output

class SAFF21(nn.Module):
    """
    Scale-Aware Feature Fusion (SAFF) Module

    Architecture:
    1. Interpolate all encoder features to bottleneck size
    2. Concatenate multi-scale features
    3. CBAM (original from paper)
    4. Parallel processing:
       - Local-Spatial Branch (DWSConv + Spatial Attention)
       - Global Branch (Transformer with ECA inside)
    5. Addition fusion
    """

    def __init__(self, encoder_channels=[16, 32, 48, 64], embed_dim=64, target_size=(8, 8, 8)):
        super(SAFF21, self).__init__()

        # Step 1: Standardize multi-scale features to same size
        self.feature_standardizer = MultiScaleFeatureStandardizer(
            encoder_channels, embed_dim, target_size
        )

        concat_channels = embed_dim * len(encoder_channels)

        # Step 2: CBAM after concatenation
        self.cbam = CBAM(concat_channels)

        # Step 3: Two parallel branches
        # Local-Spatial Branch (no residual)
        self.local_spatial_branch = LocalSpatialBranch(concat_channels, embed_dim, num_blocks=2)

        # Global Branch (with ECA inside transformer)
        self.global_branch = GlobalBranch(concat_channels, embed_dim, embed_dim,
                                          num_blocks=1, patch_size=2)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of [fused_features, e1, e2, e3]
        Returns:
            Fused bottleneck features
        """
        # Step 1: Interpolate and standardize to bottleneck size
        standardized_features = self.feature_standardizer(encoder_features)

        # Step 2: Concatenate multi-scale features
        concatenated = torch.cat(standardized_features, dim=1)

        # Step 3: CBAM (channel + spatial attention)
        attended = self.cbam(concatenated)

        # Step 4: Parallel processing
        local_spatial_features = self.local_spatial_branch(attended)
        global_features = self.global_branch(attended)

        # Step 5: Addition fusion (final output)
        output = local_spatial_features + global_features

        return output



