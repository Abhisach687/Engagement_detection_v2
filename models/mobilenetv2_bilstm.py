import torch
import torch.nn as nn
from .mobilenetv2_backbone import MobileNetV2Backbone


class MobileNetV2_BiLSTM(nn.Module):
    """
    Bidirectional LSTM variant.
    """

    def __init__(self, num_classes=4, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.backbone = MobileNetV2Backbone()
        self.lstm = nn.LSTM(
            input_size=self.backbone.out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.backbone(x)
        feat = feat.view(b, t, -1)
        lstm_out, _ = self.lstm(feat)
        final = lstm_out[:, -1, :]
        return self.fc(final)
