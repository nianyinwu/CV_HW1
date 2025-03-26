"""
Create the classification model
"""

import torch
import torch.nn as nn
import torchvision.models as models

class CBAM(nn.Module):
    """
    Define CBAM module
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial Attention Module
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward Pass
        """

        # Channel Attention Module
        avg_pool = self.mlp(self.avg_pool(x))
        max_pool = self.mlp(self.max_pool(x))
        channdel_out = self.channel_sigmoid(avg_pool+max_pool)
        x = x * channdel_out

        # Spatial Attention Module
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        spatial_out = self.spatial_sigmoid(spatial_out)
        out = x * spatial_out

        return out


def get_model():
    """
    Init the classificaion model
    """

    class_model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

    # Add CBAM after layer 4 (Stage 5)
    class_model.layer4.add_module("cbam", CBAM(channels=2048))

    # Modify classifier head
    class_model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 100)
    )
    return class_model
