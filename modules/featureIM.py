import torch
import torch.nn as nn
import torch.nn.functional as F


########################################################################
# Original Components
########################################################################

class SAB(nn.Module):
    def __init__(self, channels, r=4, reduction=4):
        super().__init__()
        self.r = r
        hidden = max(4, channels // reduction)

        self.depth_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        original_shape = x.shape[2:]
        target_shape = [max(1, s // self.r) for s in original_shape]
        x_sparse = F.adaptive_avg_pool3d(x, target_shape)

        depth_weights = self.depth_attn(x_sparse)
        spatial_weights = self.spatial_attn(x_sparse)

        alpha = torch.sigmoid(self.alpha)
        combined_weights = alpha * depth_weights + (1 - alpha) * spatial_weights

        attended = x_sparse * combined_weights
        output = F.interpolate(attended, size=original_shape, mode='trilinear', align_corners=False)

        return output


class CRB(nn.Module):
    def __init__(self, channels, r=4, reduction=2):
        super().__init__()
        hidden = max(4, channels // reduction)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_refine = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm3d(channels),
            nn.Conv3d(channels, channels, 1, bias=False)
        )

        self.local_enhance = nn.Conv3d(channels, channels, 3, padding=1,
                                       groups=max(1, min(channels, channels // 4)), bias=False)

    def forward(self, x):
        channel_weights = self.channel_attn(x)
        attended = x * channel_weights
        refined = self.spatial_refine(attended)
        enhanced = self.local_enhance(refined)
        return enhanced


class DualAttention(nn.Module):
    def __init__(self, channels, r=4, spatial_reduction=4, channel_reduction=2):
        super().__init__()
        self.spatial_stream = SAB(channels, r, spatial_reduction)
        self.channel_stream = CRB(channels, r, channel_reduction)
        self.fusion_weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, x):
        residual = x
        spatial_enhanced = self.spatial_stream(x)
        channel_enhanced = self.channel_stream(spatial_enhanced)
        fusion_weight = torch.sigmoid(self.fusion_weight)
        output = fusion_weight * channel_enhanced + (1 - fusion_weight) * x
        return output + residual


########################################################################
# Ablation: Without SAB (Spatial Attention Block)
########################################################################

class DualAttention_wo_SAB(nn.Module):
    """DualAttention without Spatial Attention Block"""
    def __init__(self, channels, r=4, spatial_reduction=4, channel_reduction=2):
        super().__init__()
        # Only channel stream, no spatial stream
        self.channel_stream = CRB(channels, r, channel_reduction)
        self.fusion_weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, x):
        residual = x
        # Skip spatial enhancement, only channel refinement
        channel_enhanced = self.channel_stream(x)
        fusion_weight = torch.sigmoid(self.fusion_weight)
        output = fusion_weight * channel_enhanced + (1 - fusion_weight) * x
        return output + residual


########################################################################
# Ablation: Without CRB (Channel Refinement Block)
########################################################################

class DualAttention_wo_CRB(nn.Module):
    """DualAttention without Channel Refinement Block"""
    def __init__(self, channels, r=4, spatial_reduction=4, channel_reduction=2):
        super().__init__()
        # Only spatial stream, no channel stream
        self.spatial_stream = SAB(channels, r, spatial_reduction)
        self.fusion_weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, x):
        residual = x
        # Only spatial enhancement, skip channel refinement
        spatial_enhanced = self.spatial_stream(x)
        fusion_weight = torch.sigmoid(self.fusion_weight)
        output = fusion_weight * spatial_enhanced + (1 - fusion_weight) * x
        return output + residual


########################################################################
# Original FIM Module
########################################################################

class FIM(nn.Module):
    def __init__(self, encoder_channels=[16, 32, 48, 64], target_channels=64, r=4):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.target_channels = target_channels

        self.dual_attention_blocks = nn.ModuleList([
            DualAttention(ch, r) for ch in encoder_channels
        ])

        self.channel_adjust = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, target_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, target_channels), target_channels)
            ) for ch in encoder_channels
        ])

        self.fusion_weights = nn.Parameter(
            torch.ones(len(encoder_channels)) / len(encoder_channels)
        )

    def forward(self, encoder_features, target_size):
        processed_features = []

        for i, features in enumerate(encoder_features):
            # Infusion Block (IB): Dual attention enhancement
            enhanced_features = self.dual_attention_blocks[i](features)

            # Size matching
            current_size = enhanced_features.shape[2:]
            if current_size[0] > target_size[0]:
                x = F.adaptive_avg_pool3d(enhanced_features, target_size)
            elif current_size[0] == target_size[0]:
                x = enhanced_features
            else:
                x = F.interpolate(enhanced_features, size=target_size, mode='trilinear', align_corners=False)

            # Channel standardization
            x = self.channel_adjust[i](x)
            processed_features.append(x)

        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_features = sum(w * feat for w, feat in zip(weights, processed_features))

        return fused_features


########################################################################
# Ablation: FIM without SAB
########################################################################

class FIM_wo_SAB(nn.Module):
    """FIM without Spatial Attention Block"""
    def __init__(self, encoder_channels=[16, 32, 48, 64], target_channels=64, r=4):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.target_channels = target_channels

        # Use DualAttention without SAB
        self.dual_attention_blocks = nn.ModuleList([
            DualAttention_wo_SAB(ch, r) for ch in encoder_channels
        ])

        self.channel_adjust = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, target_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, target_channels), target_channels)
            ) for ch in encoder_channels
        ])

        self.fusion_weights = nn.Parameter(
            torch.ones(len(encoder_channels)) / len(encoder_channels)
        )

    def forward(self, encoder_features, target_size):
        processed_features = []

        for i, features in enumerate(encoder_features):
            enhanced_features = self.dual_attention_blocks[i](features)

            current_size = enhanced_features.shape[2:]
            if current_size[0] > target_size[0]:
                x = F.adaptive_avg_pool3d(enhanced_features, target_size)
            elif current_size[0] == target_size[0]:
                x = enhanced_features
            else:
                x = F.interpolate(enhanced_features, size=target_size, mode='trilinear', align_corners=False)

            x = self.channel_adjust[i](x)
            processed_features.append(x)

        weights = F.softmax(self.fusion_weights, dim=0)
        fused_features = sum(w * feat for w, feat in zip(weights, processed_features))

        return fused_features


########################################################################
# Ablation: FIM without CRB
########################################################################

class FIM_wo_CRB(nn.Module):
    """FIM without Channel Refinement Block"""
    def __init__(self, encoder_channels=[16, 32, 48, 64], target_channels=64, r=4):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.target_channels = target_channels

        # Use DualAttention without CRB
        self.dual_attention_blocks = nn.ModuleList([
            DualAttention_wo_CRB(ch, r) for ch in encoder_channels
        ])

        self.channel_adjust = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, target_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, target_channels), target_channels)
            ) for ch in encoder_channels
        ])

        self.fusion_weights = nn.Parameter(
            torch.ones(len(encoder_channels)) / len(encoder_channels)
        )

    def forward(self, encoder_features, target_size):
        processed_features = []

        for i, features in enumerate(encoder_features):
            enhanced_features = self.dual_attention_blocks[i](features)

            current_size = enhanced_features.shape[2:]
            if current_size[0] > target_size[0]:
                x = F.adaptive_avg_pool3d(enhanced_features, target_size)
            elif current_size[0] == target_size[0]:
                x = enhanced_features
            else:
                x = F.interpolate(enhanced_features, size=target_size, mode='trilinear', align_corners=False)

            x = self.channel_adjust[i](x)
            processed_features.append(x)

        weights = F.softmax(self.fusion_weights, dim=0)
        fused_features = sum(w * feat for w, feat in zip(weights, processed_features))

        return fused_features


########################################################################
# Ablation: FIM without IB (Infusion Block)
# Process each encoder level independently - no multi-level fusion
########################################################################

class FIM_wo_IB(nn.Module):
    """
    FIM without Infusion Block (no multi-level feature aggregation)
    - SAB and CRB are applied only to features at the target level
    - No upsampling/downsampling from other encoder levels
    - Only processes the encoder feature that matches the target size
    """

    def __init__(self, encoder_channels=[16, 32, 48, 64], target_channels=64, r=4):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.target_channels = target_channels

        # DualAttention for each level (but applied independently)
        self.dual_attention_blocks = nn.ModuleList([
            DualAttention(ch, r) for ch in encoder_channels
        ])

        # Channel adjustment for each level
        self.channel_adjust = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, target_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, target_channels), target_channels)
            ) for ch in encoder_channels
        ])

    def forward(self, encoder_features, target_size):
        """
        Without IB: Only process the encoder level that matches target_size
        No multi-level feature fusion
        """
        # Find which encoder level matches the target size
        for i, features in enumerate(encoder_features):
            current_size = features.shape[2:]

            # Only process the level that matches target size
            if current_size[0] == target_size[0]:
                # Apply DualAttention to this single level only
                enhanced_features = self.dual_attention_blocks[i](features)

                # Channel standardization
                output = self.channel_adjust[i](enhanced_features)

                return output

        # Fallback: if no exact match, use the closest level
        # This shouldn't happen in normal operation
        closest_idx = min(range(len(encoder_features)),
                          key=lambda i: abs(encoder_features[i].shape[2] - target_size[0]))

        features = encoder_features[closest_idx]
        enhanced_features = self.dual_attention_blocks[closest_idx](features)

        # Resize if needed
        if features.shape[2:] != target_size:
            enhanced_features = F.interpolate(enhanced_features, size=target_size,
                                              mode='trilinear', align_corners=False)

        output = self.channel_adjust[closest_idx](enhanced_features)
        return output