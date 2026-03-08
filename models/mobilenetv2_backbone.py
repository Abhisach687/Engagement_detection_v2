import torch.nn as nn
import torchvision.models as models


class MobileNetV2Backbone(nn.Module):
    """
    Feature extractor returning a 1280-dim embedding per frame.
    """

    def __init__(self):
        super().__init__()
        model = models.mobilenet_v2(weights="DEFAULT")
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 1280

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)
