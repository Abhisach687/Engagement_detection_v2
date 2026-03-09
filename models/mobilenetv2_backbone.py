import torch
import torch.nn as nn
import torchvision.models as models

from config import BACKBONE_FRAME_CHUNK


class MobileNetV2Backbone(nn.Module):
    """
    Feature extractor returning a 1280-dim embedding per frame.
    """

    def __init__(self, frame_chunk_size: int = BACKBONE_FRAME_CHUNK):
        super().__init__()
        model = models.mobilenet_v2(weights="DEFAULT")
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 1280
        self.frame_chunk_size = frame_chunk_size

    def _forward_impl(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        if self.frame_chunk_size and x.size(0) > self.frame_chunk_size:
            outputs = []
            for start in range(0, x.size(0), self.frame_chunk_size):
                outputs.append(self._forward_impl(x[start : start + self.frame_chunk_size]))
            return torch.cat(outputs, dim=0)
        return self._forward_impl(x)
