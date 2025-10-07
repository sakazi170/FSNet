import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import L_EB, L_EB_Strided


class FrequencyDecomposer(nn.Module):
    """Full frequency decomposition with 3 frequency bands"""

    def __init__(self, channels, num_freq_bands=3):
        super(FrequencyDecomposer, self).__init__()

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
            ) for kernel_size in [1, 3, 5]
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


class ProgressiveFrequencyFusion_3Mod(nn.Module):
    """
    Adaptive 3-modality progressive fusion
    Automatically configures fusion strategy based on excluded modality
    """

    def __init__(self, target_channels, num_freq_bands=3, ex_mod='t1'):
        super(ProgressiveFrequencyFusion_3Mod, self).__init__()

        self.ex_mod = ex_mod
        self.target_channels = target_channels

        # Define all possible modality encoders - FIXED NAMING
        self.flair_freq = FrequencyDecomposer(target_channels, num_freq_bands)
        self.t2_freq = FrequencyDecomposer(target_channels, num_freq_bands)
        self.t1_freq = FrequencyDecomposer(target_channels, num_freq_bands)
        self.t1ce_freq = FrequencyDecomposer(target_channels, num_freq_bands)  # Changed from t1c_freq

        # Stage 1: Always fuse first two modalities
        self.stage1_fusion = nn.Sequential(
            nn.Conv3d(target_channels * 2 * num_freq_bands, target_channels, 1, bias=False),
            nn.GroupNorm(min(8, target_channels), target_channels),
            nn.ReLU(inplace=True)
        )

        # Stage 2: Add third modality
        self.stage2_complementarity = FrequencyComplementarityLearner(target_channels, num_freq_bands)
        self.stage2_fusion = nn.Sequential(
            nn.Conv3d(target_channels * 2, target_channels, 1, bias=False),
            nn.GroupNorm(min(8, target_channels), target_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, mod1_feat, mod2_feat, mod3_feat):
        """
        Args:
            mod1_feat, mod2_feat, mod3_feat: Features from 3 available modalities
        Returns:
            Fused features
        """
        # Stage 1: Fuse first two modalities
        mod1_freq_components = self._get_freq_components(mod1_feat, 0)
        mod2_freq_components = self._get_freq_components(mod2_feat, 1)

        # Combine frequency components
        all_freq_components = []
        for comp1, comp2 in zip(mod1_freq_components, mod2_freq_components):
            all_freq_components.extend([comp1, comp2])

        stage1_input = torch.cat(all_freq_components, dim=1)
        stage1_fused = self.stage1_fusion(stage1_input)

        # Stage 2: Add third modality
        mod3_freq_components = self._get_freq_components(mod3_feat, 2)
        stage2_complement = self.stage2_complementarity(stage1_fused, mod3_freq_components)
        stage2_input = torch.cat([stage1_fused, stage2_complement], dim=1)
        final_fused = self.stage2_fusion(stage2_input)

        return final_fused

    def _get_freq_components(self, features, mod_index):
        """
        Get frequency components based on modality position in available modalities
        """
        # Map based on excluded modality - FIXED NAMING
        if self.ex_mod == 't1':
            # Available: [t1ce, t2, flair]
            freq_decomposers = [self.t1ce_freq, self.t2_freq, self.flair_freq]
        elif self.ex_mod == 't1ce':
            # Available: [t1, t2, flair]
            freq_decomposers = [self.t1_freq, self.t2_freq, self.flair_freq]
        elif self.ex_mod == 't2':
            # Available: [t1, t1ce, flair]
            freq_decomposers = [self.t1_freq, self.t1ce_freq, self.flair_freq]
        elif self.ex_mod == 'flair':
            # Available: [t1, t1ce, t2]
            freq_decomposers = [self.t1_freq, self.t1ce_freq, self.t2_freq]
        else:
            raise ValueError(f"Invalid excluded modality: {self.ex_mod}")

        return freq_decomposers[mod_index](features)


class LightweightModalityEncoder(nn.Module):
    """Full modality encoder with 2 L_EB blocks"""

    def __init__(self, target_channels):
        super(LightweightModalityEncoder, self).__init__()

        self.encoder = nn.Sequential(
            L_EB(1, target_channels),
            L_EB_Strided(target_channels, target_channels, stride=2)
        )

    def forward(self, x):
        return self.encoder(x)


class PFMF_3mod(nn.Module):
    """
    Adaptive 3-Modality Progressive Frequency-inspired Modality Fusion Module
    Automatically adapts to any combination of 3 modalities
    """

    def __init__(self, target_channels=16, ex_mod='t1'):
        super(PFMF_3mod, self).__init__()
        self.target_channels = target_channels
        self.ex_mod = ex_mod

        # Define modality order - use 't1ce' to match your data loader
        self.modality_order = ['t1', 't1ce', 't2', 'flair']
        self.available_modalities = [mod for mod in self.modality_order if mod != ex_mod]

        print(f"PFMF configured with excluded modality: {ex_mod}")
        print(f"Available modalities: {self.available_modalities}")

        # All modality encoders - FIXED NAMING
        self.t1_encoder = LightweightModalityEncoder(target_channels)
        self.t1ce_encoder = LightweightModalityEncoder(target_channels)  # Changed from t1c_encoder
        self.t2_encoder = LightweightModalityEncoder(target_channels)
        self.flair_encoder = LightweightModalityEncoder(target_channels)

        # Adaptive progressive frequency fusion
        self.progressive_freq_fusion = ProgressiveFrequencyFusion_3Mod(
            target_channels, num_freq_bands=3, ex_mod=ex_mod
        )

        # Final output processing
        self.output_processor = nn.Sequential(
            nn.Conv3d(target_channels, target_channels, 1, bias=False),
            nn.GroupNorm(min(8, target_channels), target_channels),
            nn.GELU(),
            nn.Dropout3d(0.05)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, mod1, mod2, mod3):
        """
        Args:
            mod1, mod2, mod3: Three input modalities [B, 1, 128, 128, 128]
                              (order matches self.available_modalities)
        Returns:
            Fused features [B, target_channels, 64, 64, 64]
        """
        # Encode each modality using the correct encoder
        encoded_features = []
        for i, mod_input in enumerate([mod1, mod2, mod3]):
            mod_name = self.available_modalities[i]
            encoder = getattr(self, f"{mod_name}_encoder")
            encoded_features.append(encoder(mod_input))

        # Progressive frequency fusion
        fused_features = self.progressive_freq_fusion(*encoded_features)

        # Final processing
        output = self.output_processor(fused_features)

        return output

    def extract_frequency_components(self, mod1, mod2, mod3):
        """
        Extract processed frequency components
        Returns: dict with frequency-processed features
        """
        components = {}

        # Encode modalities
        encoded_features = []
        for i, mod_input in enumerate([mod1, mod2, mod3]):
            mod_name = self.available_modalities[i]
            encoder = getattr(self, f"{mod_name}_encoder")
            encoded_features.append(encoder(mod_input))

        # Extract frequency components for each modality
        for i, features in enumerate(encoded_features):
            mod_name = self.available_modalities[i]
            freq_components = self.progressive_freq_fusion._get_freq_components(features, i)
            for band_idx, component in enumerate(freq_components):
                components[f'{mod_name}_band{band_idx}'] = component

        return components