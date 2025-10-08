import torch
import torch.nn as nn
import os
import sys

from modules.blocks import L_EB, L_BNB, L_DB
from modules.pfmf import PFMF
from modules.ultraSB import usb
from modules.featureIM import FIM, FIM_wo_SAB, FIM_wo_CRB, FIM_wo_IB
from modules.saff import SAFF, SAFF11, SAFF21



project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


#################################################################
########################  bl   16-32-64-128 ##############################
#################################################################

class bl(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super().__init__()

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Single encoder path (combined modalities) with smaller channel sizes
        self.enc1 = L_EB(in_channels, 16)  # 4 input channels (T1, T1CE, T2, FLAIR)
        self.enc2 = L_EB(16, 32)
        self.enc3 = L_EB(32, 64)
        self.enc4 = L_EB(64, 128)

        # Bottleneck block
        self.bottleneck = L_BNB(128)

        # Decoder blocks
        self.dec1 = L_DB(256, 128)  # 128 + 128 = 256

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = L_DB(128, 64)   # 64 + 64 = 128

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = L_DB(64, 32)    # 32 + 32 = 64

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = L_DB(32, 16)    # 16 + 16 = 32

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):  # Changed signature
        # Add concatenation at the beginning
        x = torch.cat([t1, t1ce, t2, flair], dim=1)  # [B, 4, 128, 128, 128]
        # Encoder Path
        e1 = self.enc1(x)
        d1 = self.pool(e1)

        e2 = self.enc2(d1)
        d2 = self.pool(e2)

        e3 = self.enc3(d2)
        d3 = self.pool(e3)

        e4 = self.enc4(d3)

        # Bottleneck
        bottleneck = self.bottleneck(e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out

#################################################################
########################  bl2   16-32-48-64 ##############################
#################################################################

class bl2_add(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super().__init__()

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Single encoder path with new channel sizes
        self.enc1 = L_EB(in_channels, 16)  # Stays the same
        self.enc2 = L_EB(16, 32)           # Stays the same
        self.enc3 = L_EB(32, 48)           # Changed from 32→64 to 32→48
        self.enc4 = L_EB(48, 64)           # Changed from 64→128 to 48→64

        # Bottleneck block
        self.bottleneck = L_BNB(64)        # Changed from 128 to 64

        # Decoder blocks - adjusted for addition
        self.dec1 = L_DB(64, 64)           # Same channels for addition

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2),
            nn.Conv3d(48, 48, kernel_size=1)  # Ensure channel matching
        )
        self.dec2 = L_DB(48, 48)           # Same channels for addition

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose3d(48, 32, kernel_size=2, stride=2),
            nn.Conv3d(32, 32, kernel_size=1)  # Ensure channel matching
        )
        self.dec3 = L_DB(32, 32)           # Same channels for addition

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.Conv3d(16, 16, kernel_size=1)  # Ensure channel matching
        )
        self.dec4 = L_DB(16, 16)           # Same channels for addition

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):

        x = torch.cat([t1, t1ce, t2, flair], dim=1)  # [B, 4, 128, 128, 128]

        # Encoder Path
        e1 = self.enc1(x)
        d1 = self.pool(e1)

        e2 = self.enc2(d1)
        d2 = self.pool(e2)

        e3 = self.enc3(d2)
        d3 = self.pool(e3)

        e4 = self.enc4(d3)

        # Bottleneck
        bottleneck = self.bottleneck(e4)

        # Decoder Path with addition
        c5 = self.dec1(bottleneck + e4)    # Addition instead of concatenation

        up6 = self.upconv1(c5)
        c6 = self.dec2(up6 + e3)           # Addition

        up7 = self.upconv2(c6)
        c7 = self.dec3(up7 + e2)           # Addition

        up8 = self.upconv3(c7)
        c8 = self.dec4(up8 + e1)           # Addition

        out = self.final(c8)
        return out

class bl2_usb2(nn.Module):
    """Model converted from bl2_add to use USB blocks like bl4_add_usb2"""

    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super(bl2_usb2, self).__init__()

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder path with USB blocks (same channel sizes as bl2_add)
        self.enc1 = usb(in_channels, 16)  # Stays the same
        self.enc2 = usb(16, 32)           # Stays the same
        self.enc3 = usb(32, 48)           # Changed from 32→64 to 32→48
        self.enc4 = usb(48, 64)           # Changed from 64→128 to 48→64

        # Bottleneck block (keeping L_BNB as in original models)
        self.bottleneck = L_BNB(64)       # Changed from 128 to 64

        # Decoder with USB blocks - adjusted for addition
        self.dec1 = usb(64, 64)           # Same channels for addition

        self.upconv1 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)
        self.dec2 = usb(48, 48)           # Same channels for addition

        self.upconv2 = nn.ConvTranspose3d(48, 32, kernel_size=2, stride=2)
        self.dec3 = usb(32, 32)           # Same channels for addition

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = usb(16, 16)           # Same channels for addition

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        x = torch.cat([t1, t1ce, t2, flair], dim=1)  # [B, 4, 128, 128, 128]

        # Encoder Path
        e1 = self.enc1(x)
        d1 = self.pool(e1)

        e2 = self.enc2(d1)
        d2 = self.pool(e2)

        e3 = self.enc3(d2)
        d3 = self.pool(e3)

        e4 = self.enc4(d3)

        # Bottleneck
        bottleneck = self.bottleneck(e4)

        # Decoder Path with addition skip connections
        c5 = self.dec1(bottleneck + e4)    # Addition instead of concatenation

        up6 = self.upconv1(c5)
        c6 = self.dec2(up6 + e3)           # Addition

        up7 = self.upconv2(c6)
        c7 = self.dec3(up7 + e2)           # Addition

        up8 = self.upconv3(c7)
        c8 = self.dec4(up8 + e1)           # Addition

        out = self.final(c8)
        return out




class bl2_usb_pfmf_saff_fim(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super(bl2_usb_pfmf_saff_fim, self).__init__()

        self.ffm = PFMF(target_channels=16)
        self.fim_level1 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=64, r=4)
        self.fim_level2 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=48, r=4)
        self.fim_level3 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=32, r=4)
        self.fim_level4 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=16, r=4)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc1 = usb(16, 32)
        self.enc2 = usb(32, 48)
        self.enc3 = usb(48, 64)

        self.bottleneck = SAFF(encoder_channels=[16, 32, 48, 64], embed_dim=64, target_size=(8, 8, 8))

        self.dec1 = usb(64 + 64, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)
        self.dec2 = usb(48 + 48, 48)
        self.upconv2 = nn.ConvTranspose3d(48, 32, kernel_size=2, stride=2)
        self.dec3 = usb(32 + 32, 32)
        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = usb(16 + 16, 16)
        self.final_upsample = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        fused_features = self.ffm(t1, t1ce, t2, flair)  # [B, 16, 64, 64, 64]
        d0 = self.pool(fused_features)  # [B, 16, 32, 32, 32]

        e1 = self.enc1(d0)  # [B, 32, 32, 32, 32]
        d1 = self.pool(e1)  # [B, 32, 16, 16, 16]
        e2 = self.enc2(d1)  # [B, 48, 16, 16, 16]
        d2 = self.pool(e2)  # [B, 48, 8, 8, 8]
        e3 = self.enc3(d2)  # [B, 64, 8, 8, 8]

        encoder_features = [fused_features, e1, e2, e3]
        bottleneck = self.bottleneck(encoder_features)

        fim_features1 = self.fim_level1(encoder_features, (8, 8, 8))
        combined1 = torch.cat([bottleneck, fim_features1], dim=1)
        c5 = self.dec1(combined1)

        up6 = self.upconv1(c5)
        fim_features2 = self.fim_level2(encoder_features, (16, 16, 16))
        combined2 = torch.cat([up6, fim_features2], dim=1)
        c6 = self.dec2(combined2)

        up7 = self.upconv2(c6)
        fim_features3 = self.fim_level3(encoder_features, (32, 32, 32))
        combined3 = torch.cat([up7, fim_features3], dim=1)
        c7 = self.dec3(combined3)

        up8 = self.upconv3(c7)
        fim_features4 = self.fim_level4(encoder_features, (64, 64, 64))
        combined4 = torch.cat([up8, fim_features4], dim=1)
        c8 = self.dec4(combined4)

        c8_upsampled = self.final_upsample(c8)
        out = self.final(c8_upsampled)
        return out

class pfmf_saff_fim_wo_sab(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super(pfmf_saff_fim_wo_sab, self).__init__()

        self.ffm = PFMF(target_channels=16)
        self.fim_level1 = FIM_wo_SAB(encoder_channels=[16, 32, 48, 64], target_channels=64, r=4)
        self.fim_level2 = FIM_wo_SAB(encoder_channels=[16, 32, 48, 64], target_channels=48, r=4)
        self.fim_level3 = FIM_wo_SAB(encoder_channels=[16, 32, 48, 64], target_channels=32, r=4)
        self.fim_level4 = FIM_wo_SAB(encoder_channels=[16, 32, 48, 64], target_channels=16, r=4)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc1 = usb(16, 32)
        self.enc2 = usb(32, 48)
        self.enc3 = usb(48, 64)

        self.bottleneck = SAFF(encoder_channels=[16, 32, 48, 64], embed_dim=64, target_size=(8, 8, 8))

        self.dec1 = usb(64 + 64, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)
        self.dec2 = usb(48 + 48, 48)
        self.upconv2 = nn.ConvTranspose3d(48, 32, kernel_size=2, stride=2)
        self.dec3 = usb(32 + 32, 32)
        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = usb(16 + 16, 16)
        self.final_upsample = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        fused_features = self.ffm(t1, t1ce, t2, flair)  # [B, 16, 64, 64, 64]
        d0 = self.pool(fused_features)  # [B, 16, 32, 32, 32]

        e1 = self.enc1(d0)  # [B, 32, 32, 32, 32]
        d1 = self.pool(e1)  # [B, 32, 16, 16, 16]
        e2 = self.enc2(d1)  # [B, 48, 16, 16, 16]
        d2 = self.pool(e2)  # [B, 48, 8, 8, 8]
        e3 = self.enc3(d2)  # [B, 64, 8, 8, 8]

        encoder_features = [fused_features, e1, e2, e3]
        bottleneck = self.bottleneck(encoder_features)

        fim_features1 = self.fim_level1(encoder_features, (8, 8, 8))
        combined1 = torch.cat([bottleneck, fim_features1], dim=1)
        c5 = self.dec1(combined1)

        up6 = self.upconv1(c5)
        fim_features2 = self.fim_level2(encoder_features, (16, 16, 16))
        combined2 = torch.cat([up6, fim_features2], dim=1)
        c6 = self.dec2(combined2)

        up7 = self.upconv2(c6)
        fim_features3 = self.fim_level3(encoder_features, (32, 32, 32))
        combined3 = torch.cat([up7, fim_features3], dim=1)
        c7 = self.dec3(combined3)

        up8 = self.upconv3(c7)
        fim_features4 = self.fim_level4(encoder_features, (64, 64, 64))
        combined4 = torch.cat([up8, fim_features4], dim=1)
        c8 = self.dec4(combined4)

        c8_upsampled = self.final_upsample(c8)
        out = self.final(c8_upsampled)
        return out

class pfmf_saff_fim_wo_crb(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super(pfmf_saff_fim_wo_crb, self).__init__()

        self.ffm = PFMF(target_channels=16)
        self.fim_level1 = FIM_wo_CRB(encoder_channels=[16, 32, 48, 64], target_channels=64, r=4)
        self.fim_level2 = FIM_wo_CRB(encoder_channels=[16, 32, 48, 64], target_channels=48, r=4)
        self.fim_level3 = FIM_wo_CRB(encoder_channels=[16, 32, 48, 64], target_channels=32, r=4)
        self.fim_level4 = FIM_wo_CRB(encoder_channels=[16, 32, 48, 64], target_channels=16, r=4)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc1 = usb(16, 32)
        self.enc2 = usb(32, 48)
        self.enc3 = usb(48, 64)

        self.bottleneck = SAFF(encoder_channels=[16, 32, 48, 64], embed_dim=64, target_size=(8, 8, 8))

        self.dec1 = usb(64 + 64, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)
        self.dec2 = usb(48 + 48, 48)
        self.upconv2 = nn.ConvTranspose3d(48, 32, kernel_size=2, stride=2)
        self.dec3 = usb(32 + 32, 32)
        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = usb(16 + 16, 16)
        self.final_upsample = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        fused_features = self.ffm(t1, t1ce, t2, flair)  # [B, 16, 64, 64, 64]
        d0 = self.pool(fused_features)  # [B, 16, 32, 32, 32]

        e1 = self.enc1(d0)  # [B, 32, 32, 32, 32]
        d1 = self.pool(e1)  # [B, 32, 16, 16, 16]
        e2 = self.enc2(d1)  # [B, 48, 16, 16, 16]
        d2 = self.pool(e2)  # [B, 48, 8, 8, 8]
        e3 = self.enc3(d2)  # [B, 64, 8, 8, 8]

        encoder_features = [fused_features, e1, e2, e3]
        bottleneck = self.bottleneck(encoder_features)

        fim_features1 = self.fim_level1(encoder_features, (8, 8, 8))
        combined1 = torch.cat([bottleneck, fim_features1], dim=1)
        c5 = self.dec1(combined1)

        up6 = self.upconv1(c5)
        fim_features2 = self.fim_level2(encoder_features, (16, 16, 16))
        combined2 = torch.cat([up6, fim_features2], dim=1)
        c6 = self.dec2(combined2)

        up7 = self.upconv2(c6)
        fim_features3 = self.fim_level3(encoder_features, (32, 32, 32))
        combined3 = torch.cat([up7, fim_features3], dim=1)
        c7 = self.dec3(combined3)

        up8 = self.upconv3(c7)
        fim_features4 = self.fim_level4(encoder_features, (64, 64, 64))
        combined4 = torch.cat([up8, fim_features4], dim=1)
        c8 = self.dec4(combined4)

        c8_upsampled = self.final_upsample(c8)
        out = self.final(c8_upsampled)
        return out

class pfmf_saff_fim_wo_ib(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super(pfmf_saff_fim_wo_ib, self).__init__()

        self.ffm = PFMF(target_channels=16)
        self.fim_level1 = FIM_wo_IB(encoder_channels=[16, 32, 48, 64], target_channels=64, r=4)
        self.fim_level2 = FIM_wo_IB(encoder_channels=[16, 32, 48, 64], target_channels=48, r=4)
        self.fim_level3 = FIM_wo_IB(encoder_channels=[16, 32, 48, 64], target_channels=32, r=4)
        self.fim_level4 = FIM_wo_IB(encoder_channels=[16, 32, 48, 64], target_channels=16, r=4)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc1 = usb(16, 32)
        self.enc2 = usb(32, 48)
        self.enc3 = usb(48, 64)

        self.bottleneck = SAFF(encoder_channels=[16, 32, 48, 64], embed_dim=64, target_size=(8, 8, 8))

        self.dec1 = usb(64 + 64, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)
        self.dec2 = usb(48 + 48, 48)
        self.upconv2 = nn.ConvTranspose3d(48, 32, kernel_size=2, stride=2)
        self.dec3 = usb(32 + 32, 32)
        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = usb(16 + 16, 16)
        self.final_upsample = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        fused_features = self.ffm(t1, t1ce, t2, flair)  # [B, 16, 64, 64, 64]
        d0 = self.pool(fused_features)  # [B, 16, 32, 32, 32]

        e1 = self.enc1(d0)  # [B, 32, 32, 32, 32]
        d1 = self.pool(e1)  # [B, 32, 16, 16, 16]
        e2 = self.enc2(d1)  # [B, 48, 16, 16, 16]
        d2 = self.pool(e2)  # [B, 48, 8, 8, 8]
        e3 = self.enc3(d2)  # [B, 64, 8, 8, 8]

        encoder_features = [fused_features, e1, e2, e3]
        bottleneck = self.bottleneck(encoder_features)

        fim_features1 = self.fim_level1(encoder_features, (8, 8, 8))
        combined1 = torch.cat([bottleneck, fim_features1], dim=1)
        c5 = self.dec1(combined1)

        up6 = self.upconv1(c5)
        fim_features2 = self.fim_level2(encoder_features, (16, 16, 16))
        combined2 = torch.cat([up6, fim_features2], dim=1)
        c6 = self.dec2(combined2)

        up7 = self.upconv2(c6)
        fim_features3 = self.fim_level3(encoder_features, (32, 32, 32))
        combined3 = torch.cat([up7, fim_features3], dim=1)
        c7 = self.dec3(combined3)

        up8 = self.upconv3(c7)
        fim_features4 = self.fim_level4(encoder_features, (64, 64, 64))
        combined4 = torch.cat([up8, fim_features4], dim=1)
        c8 = self.dec4(combined4)

        c8_upsampled = self.final_upsample(c8)
        out = self.final(c8_upsampled)
        return out

class bl2_usb_pfmf_saff11_fim(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super(bl2_usb_pfmf_saff11_fim, self).__init__()

        self.ffm = PFMF(target_channels=16)
        self.fim_level1 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=64, r=4)
        self.fim_level2 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=48, r=4)
        self.fim_level3 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=32, r=4)
        self.fim_level4 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=16, r=4)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc1 = usb(16, 32)
        self.enc2 = usb(32, 48)
        self.enc3 = usb(48, 64)

        self.bottleneck = SAFF11(encoder_channels=[16, 32, 48, 64], embed_dim=64, target_size=(8, 8, 8))

        self.dec1 = usb(64 + 64, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)
        self.dec2 = usb(48 + 48, 48)
        self.upconv2 = nn.ConvTranspose3d(48, 32, kernel_size=2, stride=2)
        self.dec3 = usb(32 + 32, 32)
        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = usb(16 + 16, 16)
        self.final_upsample = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        fused_features = self.ffm(t1, t1ce, t2, flair)  # [B, 16, 64, 64, 64]
        d0 = self.pool(fused_features)  # [B, 16, 32, 32, 32]

        e1 = self.enc1(d0)  # [B, 32, 32, 32, 32]
        d1 = self.pool(e1)  # [B, 32, 16, 16, 16]
        e2 = self.enc2(d1)  # [B, 48, 16, 16, 16]
        d2 = self.pool(e2)  # [B, 48, 8, 8, 8]
        e3 = self.enc3(d2)  # [B, 64, 8, 8, 8]

        encoder_features = [fused_features, e1, e2, e3]
        bottleneck = self.bottleneck(encoder_features)

        fim_features1 = self.fim_level1(encoder_features, (8, 8, 8))
        combined1 = torch.cat([bottleneck, fim_features1], dim=1)
        c5 = self.dec1(combined1)

        up6 = self.upconv1(c5)
        fim_features2 = self.fim_level2(encoder_features, (16, 16, 16))
        combined2 = torch.cat([up6, fim_features2], dim=1)
        c6 = self.dec2(combined2)

        up7 = self.upconv2(c6)
        fim_features3 = self.fim_level3(encoder_features, (32, 32, 32))
        combined3 = torch.cat([up7, fim_features3], dim=1)
        c7 = self.dec3(combined3)

        up8 = self.upconv3(c7)
        fim_features4 = self.fim_level4(encoder_features, (64, 64, 64))
        combined4 = torch.cat([up8, fim_features4], dim=1)
        c8 = self.dec4(combined4)

        c8_upsampled = self.final_upsample(c8)
        out = self.final(c8_upsampled)
        return out

class bl2_usb_pfmf_saff21_fim(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super(bl2_usb_pfmf_saff21_fim, self).__init__()

        self.ffm = PFMF(target_channels=16)
        self.fim_level1 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=64, r=4)
        self.fim_level2 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=48, r=4)
        self.fim_level3 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=32, r=4)
        self.fim_level4 = FIM(encoder_channels=[16, 32, 48, 64], target_channels=16, r=4)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc1 = usb(16, 32)
        self.enc2 = usb(32, 48)
        self.enc3 = usb(48, 64)

        self.bottleneck = SAFF21(encoder_channels=[16, 32, 48, 64], embed_dim=64, target_size=(8, 8, 8))

        self.dec1 = usb(64 + 64, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)
        self.dec2 = usb(48 + 48, 48)
        self.upconv2 = nn.ConvTranspose3d(48, 32, kernel_size=2, stride=2)
        self.dec3 = usb(32 + 32, 32)
        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = usb(16 + 16, 16)
        self.final_upsample = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        fused_features = self.ffm(t1, t1ce, t2, flair)  # [B, 16, 64, 64, 64]
        d0 = self.pool(fused_features)  # [B, 16, 32, 32, 32]

        e1 = self.enc1(d0)  # [B, 32, 32, 32, 32]
        d1 = self.pool(e1)  # [B, 32, 16, 16, 16]
        e2 = self.enc2(d1)  # [B, 48, 16, 16, 16]
        d2 = self.pool(e2)  # [B, 48, 8, 8, 8]
        e3 = self.enc3(d2)  # [B, 64, 8, 8, 8]

        encoder_features = [fused_features, e1, e2, e3]
        bottleneck = self.bottleneck(encoder_features)

        fim_features1 = self.fim_level1(encoder_features, (8, 8, 8))
        combined1 = torch.cat([bottleneck, fim_features1], dim=1)
        c5 = self.dec1(combined1)

        up6 = self.upconv1(c5)
        fim_features2 = self.fim_level2(encoder_features, (16, 16, 16))
        combined2 = torch.cat([up6, fim_features2], dim=1)
        c6 = self.dec2(combined2)

        up7 = self.upconv2(c6)
        fim_features3 = self.fim_level3(encoder_features, (32, 32, 32))
        combined3 = torch.cat([up7, fim_features3], dim=1)
        c7 = self.dec3(combined3)

        up8 = self.upconv3(c7)
        fim_features4 = self.fim_level4(encoder_features, (64, 64, 64))
        combined4 = torch.cat([up8, fim_features4], dim=1)
        c8 = self.dec4(combined4)

        c8_upsampled = self.final_upsample(c8)
        out = self.final(c8_upsampled)
        return out





#################################################################
########################  bl3   24-48-72-96  ##############################
#################################################################

class bl3_add(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super(bl3_add, self).__init__()

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder path
        self.enc1 = L_EB(in_channels, 24)
        self.enc2 = L_EB(24, 48)
        self.enc3 = L_EB(48, 72)
        self.enc4 = L_EB(72, 96)

        # Bottleneck
        self.bottleneck = L_BNB(96)

        # Decoder path with addition
        self.dec1 = L_DB(96, 96)
        # 1x1 convs ensure channel matching for addition
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose3d(96, 72, kernel_size=2, stride=2),
            nn.Conv3d(72, 72, kernel_size=1)
        )

        self.dec2 = L_DB(72, 72)
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose3d(72, 48, kernel_size=2, stride=2),
            nn.Conv3d(48, 48, kernel_size=1)
        )

        self.dec3 = L_DB(48, 48)
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2),
            nn.Conv3d(24, 24, kernel_size=1)
        )

        self.dec4 = L_DB(24, 24)
        self.final = nn.Conv3d(24, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):

        x = torch.cat([t1, t1ce, t2, flair], dim=1)  # [B, 4, 128, 128, 128]

        # Encoder Path
        e1 = self.enc1(x)
        d1 = self.pool(e1)

        e2 = self.enc2(d1)
        d2 = self.pool(e2)

        e3 = self.enc3(d2)
        d3 = self.pool(e3)

        e4 = self.enc4(d3)

        # Bottleneck
        bottleneck = self.bottleneck(e4)

        # Decoder Path with addition
        c5 = self.dec1(bottleneck + e4)

        up6 = self.upconv1(c5)
        c6 = self.dec2(up6 + e3)

        up7 = self.upconv2(c6)
        c7 = self.dec3(up7 + e2)

        up8 = self.upconv3(c7)
        c8 = self.dec4(up8 + e1)

        out = self.final(c8)
        return out


#################################################################
########################  bl4   16-32-64-96  ##############################
#################################################################

class bl4_add(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super(bl4_add, self).__init__()

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder path with 16→32→64→96 progression
        self.enc1 = L_EB(in_channels, 16)  # Start with 16 channels
        self.enc2 = L_EB(16, 32)  # Double to 32
        self.enc3 = L_EB(32, 64)  # Double to 64
        self.enc4 = L_EB(64, 96)  # 1.5x to 96

        # Bottleneck block
        self.bottleneck = L_BNB(96)

        # Decoder with addition skip connections
        self.dec1 = L_DB(96, 96)  # Same channels for addition
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose3d(96, 64, kernel_size=2, stride=2),
            nn.Conv3d(64, 64, kernel_size=1)  # Ensure channel matching
        )

        self.dec2 = L_DB(64, 64)  # Same channels for addition
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.Conv3d(32, 32, kernel_size=1)  # Ensure channel matching
        )

        self.dec3 = L_DB(32, 32)  # Same channels for addition
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.Conv3d(16, 16, kernel_size=1)  # Ensure channel matching
        )

        self.dec4 = L_DB(16, 16)  # Same channels for addition

        # Final layer
        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):

        x = torch.cat([t1, t1ce, t2, flair], dim=1)  # [B, 4, 128, 128, 128]

        # Encoder Path
        e1 = self.enc1(x)  # [B, 16, 128, 128, 128]
        d1 = self.pool(e1)  # [B, 16, 64, 64, 64]

        e2 = self.enc2(d1)  # [B, 32, 64, 64, 64]
        d2 = self.pool(e2)  # [B, 32, 32, 32, 32]

        e3 = self.enc3(d2)  # [B, 64, 32, 32, 32]
        d3 = self.pool(e3)  # [B, 64, 16, 16, 16]

        e4 = self.enc4(d3)  # [B, 96, 16, 16, 16]

        # Bottleneck
        bottleneck = self.bottleneck(e4)  # [B, 96, 16, 16, 16]

        # Decoder Path with addition
        c5 = self.dec1(bottleneck + e4)  # [B, 96, 16, 16, 16]

        up6 = self.upconv1(c5)  # [B, 64, 32, 32, 32]
        c6 = self.dec2(up6 + e3)  # [B, 64, 32, 32, 32]

        up7 = self.upconv2(c6)  # [B, 32, 64, 64, 64]
        c7 = self.dec3(up7 + e2)  # [B, 32, 64, 64, 64]

        up8 = self.upconv3(c7)  # [B, 16, 128, 128, 128]
        c8 = self.dec4(up8 + e1)  # [B, 16, 128, 128, 128]

        out = self.final(c8)  # [B, 4, 128, 128, 128]
        return out

class bl4_usb2(nn.Module):
    """Model 2: USB encoder blocks + USB decoder blocks"""

    def __init__(self, img_h, img_w, img_d, in_channels=4, num_classes=4):
        super(bl4_usb2, self).__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder path with USB blocks
        self.enc1 = usb(in_channels, 16)
        self.enc2 = usb(16, 32)
        self.enc3 = usb(32, 64)
        self.enc4 = usb(64, 96)

        # Bottleneck block
        self.bottleneck = L_BNB(96)

        # Decoder with USB blocks
        self.dec1 = usb(96, 96)
        self.upconv1 = nn.ConvTranspose3d(96, 64, kernel_size=2, stride=2)

        self.dec2 = usb(64, 64)
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)

        self.dec3 = usb(32, 32)
        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)

        self.dec4 = usb(16, 16)
        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        x = torch.cat([t1, t1ce, t2, flair], dim=1)

        # Encoder Path
        e1 = self.enc1(x)
        d1 = self.pool(e1)

        e2 = self.enc2(d1)
        d2 = self.pool(e2)

        e3 = self.enc3(d2)
        d3 = self.pool(e3)

        e4 = self.enc4(d3)

        # Bottleneck
        bottleneck = self.bottleneck(e4)

        # Decoder Path with addition skip connections
        c5 = self.dec1(bottleneck + e4)

        up6 = self.upconv1(c5)
        c6 = self.dec2(up6 + e3)

        up7 = self.upconv2(c6)
        c7 = self.dec3(up7 + e2)

        up8 = self.upconv3(c7)
        c8 = self.dec4(up8 + e1)

        out = self.final(c8)
        return out

