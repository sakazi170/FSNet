import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import L_EB, L_EB_Strided


class FrequencyDecomposition(nn.Module):
    """Full frequency decomposition with 3 frequency bands"""

    def __init__(self, channels, num_freq_bands=3):
        super(FrequencyDecomposition, self).__init__()

        self.freq_selectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=1, groups=max(1, channels // 8), bias=False),
                nn.GroupNorm(min(8, channels), channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, 1, bias=False),
                nn.Sigmoid()
            ) for _ in range(num_freq_bands)
        ])

        self.band_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
                nn.GroupNorm(min(8, channels), channels),
                nn.ReLU(inplace=True)
            ) for kernel_size in [1, 3, 5]  # Full 3 frequency bands
        ])

        self.residual_weights = nn.Parameter(torch.ones(num_freq_bands) * 0.1)

    def forward(self, x):
        freq_components = []
        for i, (selector, processor) in enumerate(zip(self.freq_selectors, self.band_processors)):
            freq_mask = selector(x)
            freq_component = processor(x * freq_mask)
            residual_weight = torch.sigmoid(self.residual_weights[i])
            stable_component = freq_component + residual_weight * x
            freq_components.append(stable_component)
        return freq_components


class FrequencyComplementarityLearner(nn.Module):
    """Full complementarity learning module"""

    def __init__(self, channels, num_freq_bands=3):
        super(FrequencyComplementarityLearner, self).__init__()

        self.missing_freq_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, num_freq_bands, 1),
            nn.Sigmoid()
        )

        self.complementarity_selector = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, 1, bias=False),
                nn.GroupNorm(min(4, channels), channels)
            ) for _ in range(num_freq_bands)
        ])

        self.frequency_combiner = nn.Sequential(
            nn.Conv3d(channels * num_freq_bands, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, current_fused, new_modality_freq_components):
        missing_weights = self.missing_freq_analyzer(current_fused)

        complementary_components = []
        for i, (component, selector) in enumerate(zip(new_modality_freq_components, self.complementarity_selector)):
            weight = missing_weights[:, i:i + 1, :, :, :]
            selected = selector(component) * weight
            complementary_components.append(selected)

        combined_complement = torch.cat(complementary_components, dim=1)
        final_complement = self.frequency_combiner(combined_complement)
        return final_complement


class ProgressiveFrequencyFusion_Alt2(nn.Module):
    """Progressive fusion: (T1ce + FLAIR) → + T2 → + T1"""

    def __init__(self, target_channels, num_freq_bands=3):
        super(ProgressiveFrequencyFusion_Alt2, self).__init__()

        # Stage 1: T1ce + FLAIR (separate modules)
        self.stage1_t1ce_freq = FrequencyDecomposition(target_channels, num_freq_bands)
        self.stage1_flair_freq = FrequencyDecomposition(target_channels, num_freq_bands)
        self.stage1_fusion = nn.Sequential(
            nn.Conv3d(target_channels * 2 * num_freq_bands, target_channels, 1, bias=False),
            nn.GroupNorm(min(8, target_channels), target_channels),
            nn.ReLU(inplace=True)
        )

        # Stage 2: Add T2 (separate modules)
        self.stage2_t2_freq = FrequencyDecomposition(target_channels, num_freq_bands)
        self.stage2_complementarity = FrequencyComplementarityLearner(target_channels, num_freq_bands)
        self.stage2_fusion = nn.Sequential(
            nn.Conv3d(target_channels * 2, target_channels, 1, bias=False),
            nn.GroupNorm(min(8, target_channels), target_channels),
            nn.ReLU(inplace=True)
        )

        # Stage 3: Add T1 (separate modules)
        self.stage3_t1_freq = FrequencyDecomposition(target_channels, num_freq_bands)
        self.stage3_complementarity = FrequencyComplementarityLearner(target_channels, num_freq_bands)
        self.stage3_fusion = nn.Sequential(
            nn.Conv3d(target_channels * 2, target_channels, 1, bias=False),
            nn.GroupNorm(min(8, target_channels), target_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, t1ce_feat, flair_feat, t2_feat, t1_feat):
        # Stage 1: T1ce + FLAIR frequency complementarity
        t1ce_freq_components = self.stage1_t1ce_freq(t1ce_feat)
        flair_freq_components = self.stage1_flair_freq(flair_feat)

        # Combine frequency components (2 modalities * 3 freq_bands = 6 tensors)
        all_freq_components = []
        for t1ce_comp, flair_comp in zip(t1ce_freq_components, flair_freq_components):
            all_freq_components.extend([t1ce_comp, flair_comp])

        stage1_input = torch.cat(all_freq_components, dim=1)  # 6 * 16 = 96 channels
        stage1_fused = self.stage1_fusion(stage1_input)

        # Stage 2: Add T2
        t2_freq_components = self.stage2_t2_freq(t2_feat)
        stage2_complement = self.stage2_complementarity(stage1_fused, t2_freq_components)
        stage2_input = torch.cat([stage1_fused, stage2_complement], dim=1)
        stage2_fused = self.stage2_fusion(stage2_input)

        # Stage 3: Add T1
        t1_freq_components = self.stage3_t1_freq(t1_feat)
        stage3_complement = self.stage3_complementarity(stage2_fused, t1_freq_components)
        stage3_input = torch.cat([stage2_fused, stage3_complement], dim=1)
        final_fused = self.stage3_fusion(stage3_input)

        return final_fused


class LightweightModalityEncoder(nn.Module):
    """Full modality encoder with 2 L_EB blocks - stride second approach"""

    def __init__(self, target_channels):
        super(LightweightModalityEncoder, self).__init__()

        self.encoder = nn.Sequential(
            L_EB(1, target_channels),  # 128³ → 128³, 1 → 16 channels
            L_EB_Strided(target_channels, target_channels, stride=2)  # 128³ → 64³, 16 → 16 channels
        )

    def forward(self, x):
        return self.encoder(x)


class PFMF2(nn.Module):
    """
    Alternative 2: Progressive Frequency-inspired Modality Fusion
    Fusion Order: (T1ce + FLAIR) → + T2 → + T1
    - 3 frequency bands (1x1, 3x3, 5x5)
    - Full frequency complementarity learning
    - Separate modules for each progressive stage
    - 2 L_EB blocks per modality (with stride in second one)
    """

    def __init__(self, target_channels=16):
        super(PFMF2, self).__init__()
        self.target_channels = target_channels

        # Full modality encoders (2 L_EB each, second one strided)
        self.t1ce_encoder = LightweightModalityEncoder(target_channels)
        self.flair_encoder = LightweightModalityEncoder(target_channels)
        self.t2_encoder = LightweightModalityEncoder(target_channels)
        self.t1_encoder = LightweightModalityEncoder(target_channels)

        # Progressive frequency fusion: (T1ce + FLAIR) → + T2 → + T1
        self.progressive_freq_fusion = ProgressiveFrequencyFusion_Alt2(target_channels)

        # Final output processing
        self.output_processor = nn.Sequential(
            nn.Conv3d(target_channels, target_channels, 1, bias=False),
            nn.GroupNorm(min(8, target_channels), target_channels),
            nn.GELU(),
            nn.Dropout3d(0.05)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, I_flair, I_t2, I_t1, I_t1c):
        """
        Args:
            I_flair, I_t2, I_t1, I_t1c: Input modalities [B, 1, 128, 128, 128]
        Returns:
            Fused features [B, target_channels, 64, 64, 64]
        """
        # Full modality-specific encoding (128³ → 64³)
        t1ce_features = self.t1ce_encoder(I_t1c)
        flair_features = self.flair_encoder(I_flair)
        t2_features = self.t2_encoder(I_t2)
        t1_features = self.t1_encoder(I_t1)

        # Progressive frequency-domain fusion: (T1ce + FLAIR) → + T2 → + T1
        fused_features = self.progressive_freq_fusion(
            t1ce_features, flair_features, t2_features, t1_features
        )

        # Final processing
        output = self.output_processor(fused_features)

        return output