import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Feature Extractor (ViT-B/8)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = models.vit_b_8(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove classification head
    
    def forward(self, x):
        return self.vit(x)

# Low-pass Feature Enhancement (LFE)
class LFE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.low_pass_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False)
        nn.init.constant_(self.low_pass_conv.weight, 1/25)  # Initialize as averaging filter
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        low_freq = self.low_pass_conv(x)
        attn = self.attention(low_freq)
        return low_freq * attn

# Illumination-aware Feature Enhancement (IFE)
class IFE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x, illumination_map):
        illum_features = self.conv(illumination_map)
        attn = self.spatial_attn(illum_features)
        return x * attn

# Adaptive Feature Fusion (AFF)
class AFF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        fused = x1 + x2
        ca = self.channel_attn(fused)
        sa = self.spatial_attn(fused)
        return fused * ca * sa

# Anomaly Detector Network
class AnomalyDetector(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.lfe = LFE(channels)
        self.ife = IFE(channels)
        self.aff = AFF(channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1)
        )
    
    def forward(self, well_lit, low_light, illumination_map):
        f_wl = self.feature_extractor(well_lit)
        f_ll = self.feature_extractor(low_light)
        
        f_lfe = self.lfe(f_ll)
        f_ife = self.ife(f_ll, illumination_map)
        
        f_fused = self.aff(f_lfe, f_ife)
        f_pred = self.decoder(f_fused)
        
        anomaly_map = torch.norm(f_wl - f_pred, p=2, dim=1, keepdim=True)
        return anomaly_map
