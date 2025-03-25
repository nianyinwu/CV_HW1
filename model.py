import torch
import torch.nn as nn
import torchvision.models as models

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.MaxPool = nn.AdaptiveMaxPool2d(1)
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.MLP = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.Sigmoid_C = nn.Sigmoid()

        # Spatial Attention Module
        self.Conv_S = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.Sigmoid_S = nn.Sigmoid()

    def forward(self, x):
        avgPool = self.MLP(self.AvgPool(x))
        maxPool = self.MLP(self.MaxPool(x))
        C_out = self.Sigmoid_C(avgPool+maxPool)
        x = x * C_out

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        S_out = self.Conv_S(torch.cat([avg_out, max_out], dim=1))
        S_out = self.Sigmoid_S(S_out)
        out = x * S_out

        return out


def model():
    model = models.resnext50_32x4d(
        weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

    # Add CBAM after layer4
    model.layer4.add_module("cbam", CBAM(channels=2048))

    # Modify classifier head
    model.fc = nn.Sequential(
        nn.Linear(2048, 100),
        # nn.BatchNorm1d(1024),
        # nn.SiLU(),
        # nn.Dropout(0.3),
        # nn.Linear(1024, 512),
        # nn.BatchNorm1d(512),
        # nn.SiLU(),
        # nn.Dropout(0.3),
        # nn.Linear(512, 100)
    )
    return model
