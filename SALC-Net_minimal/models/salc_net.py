from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class ScaleAdaptiveEfficientDynamicPyramid(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        reduced = channels // reduction
        self.branch_small = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.Conv2d(reduced, reduced, 3, padding=1, groups=reduced, bias=False),
            nn.Conv2d(reduced, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.GELU(),
        )
        self.branch_medium = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.Conv2d(reduced, reduced, 3, padding=2, dilation=2, groups=reduced, bias=False),
            nn.Conv2d(reduced, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.GELU(),
        )
        self.scale_gate = nn.Sequential(
            nn.Conv2d(reduced * 2, reduced, 1, bias=True),
            nn.BatchNorm2d(reduced),
            nn.GELU(),
            nn.Conv2d(reduced, 2, 1, bias=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        feat_small = self.branch_small(x)
        feat_medium = self.branch_medium(x)
        gate = self.scale_gate(torch.cat([feat_small, feat_medium], dim=1))
        gate = F.softmax(gate, dim=1)
        fused = gate[:, 0:1] * feat_small + gate[:, 1:2] * feat_medium
        return self.fusion(fused) + x


class LinearCrossScaleFusion(nn.Module):
    def __init__(self, high_dim, low_dim, out_dim, reduction=4):
        super().__init__()
        self.high_proj = nn.Sequential(nn.Conv2d(high_dim, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim))
        self.low_proj = nn.Sequential(nn.Conv2d(low_dim, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim))
        self.high_guidance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(out_dim // reduction, out_dim, 1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 5, padding=2, groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(out_dim // reduction, out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, high_feat, low_feat):
        height, width = low_feat.shape[2:]
        high_feat = F.interpolate(high_feat, size=(height, width), mode="bilinear", align_corners=False)
        high_feat = self.high_proj(high_feat)
        low_feat = self.low_proj(low_feat)
        low_guided = low_feat * self.high_guidance(high_feat)
        fused = high_feat + low_guided
        fused = fused * self.spatial_attention(fused) * self.channel_attention(fused)
        return fused + low_feat


class MobileNetV2Backbone(nn.Module):
    def __init__(self, downsample_factor=16):
        super().__init__()
        model = mobilenet_v2(weights=None)
        self.features = model.features[:-1]
        self.down_idx = [2, 4, 7, 14]
        total_idx = len(self.features)
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
        else:
            raise ValueError("downsample_factor must be 8 or 16")

    def _nostride_dilate(self, module, dilate):
        if module.__class__.__name__.find("Conv") == -1:
            return
        if module.stride == (2, 2):
            module.stride = (1, 1)
            if module.kernel_size == (3, 3):
                module.dilation = (dilate // 2, dilate // 2)
                module.padding = (dilate // 2, dilate // 2)
        elif module.kernel_size == (3, 3):
            module.dilation = (dilate, dilate)
            module.padding = (dilate, dilate)

    def forward(self, x):
        low = self.features[:4](x)
        high = self.features[4:](low)
        return low, high


class SALCNet(nn.Module):
    def __init__(self, num_classes=2, downsample_factor=16):
        super().__init__()
        self.backbone = MobileNetV2Backbone(downsample_factor=downsample_factor)
        self.sa_edp = ScaleAdaptiveEfficientDynamicPyramid(320, reduction=2)
        self.high_reduce = nn.Sequential(nn.Conv2d(320, 160, 1, bias=False), nn.BatchNorm2d(160), nn.GELU())
        self.low_reduce = nn.Sequential(nn.Conv2d(24, 40, 1, bias=False), nn.BatchNorm2d(40), nn.GELU())
        self.lcsf = LinearCrossScaleFusion(160, 40, 160, reduction=4)
        self.decoder = nn.Sequential(
            nn.Conv2d(200, 160, 3, padding=1, groups=40, bias=False),
            nn.Conv2d(160, 160, 1, bias=False),
            nn.BatchNorm2d(160),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(160, 160, 3, padding=1, groups=40, bias=False),
            nn.Conv2d(160, 160, 1, bias=False),
            nn.BatchNorm2d(160),
            nn.GELU(),
        )
        self.cls_head = nn.Conv2d(160, num_classes, 1)

    def forward(self, x):
        height, width = x.shape[2:]
        low, high = self.backbone(x)
        high = self.sa_edp(high)
        high = self.high_reduce(high)
        low = self.low_reduce(low)
        fused = self.lcsf(high, low)
        fused = torch.cat([fused, low], dim=1)
        logits = self.cls_head(self.decoder(fused))
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


class DeepLab(nn.Module):
    def __init__(self, num_classes=2, backbone="mobilenet", pretrained=False, downsample_factor=16, **kwargs):
        super().__init__()
        if backbone != "mobilenet":
            raise ValueError("This release uses MobileNetV2 only.")
        if pretrained:
            raise ValueError("The reported experiments used random initialization.")
        self.model = SALCNet(num_classes=num_classes, downsample_factor=downsample_factor)

    def forward(self, x):
        return self.model(x)
