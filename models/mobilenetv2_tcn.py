import torch
import torch.nn as nn
from .mobilenetv2_backbone import MobileNetV2Backbone


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class MobileNetV2_TCN(nn.Module):
    """
    Temporal Conv Net on top of MobileNetV2 frame embeddings.
    """

    def __init__(self, num_classes=4, num_heads: int = 1):
        super().__init__()
        self.backbone = MobileNetV2Backbone()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.tcn = nn.Sequential(
            TemporalBlock(1280, 512, 1),
            TemporalBlock(512, 256, 2),
            TemporalBlock(256, 128, 4),
        )
        self.fc = nn.Linear(128, num_classes * num_heads)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)  # (B*T, 1280)
        feats = feats.view(b, t, -1)  # (B, T, 1280)
        feats = feats.permute(0, 2, 1)  # (B, 1280, T)
        out = self.tcn(feats)  # (B, 128, T)
        out = out[:, :, -1]
        logits = self.fc(out)
        if self.num_heads == 1:
            return logits
        return logits.view(b, self.num_heads, self.num_classes)
