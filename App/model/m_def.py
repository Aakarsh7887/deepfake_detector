import torch
import torch.nn as nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MyModel, self).__init__()

        base_model = models.resnext50_32x4d(weights=None)

        # ✅ SAME NAME AS TRAINING
        self.model = nn.Sequential(*list(base_model.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d(1)

        # ✅ LSTM (IMPORTANT)
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=2048,
            num_layers=1,
            batch_first=True,
            bias = False
        )

        # ✅ SAME NAME
        self.linear1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        # x: (1, frames, C, H, W)

        batch, frames, c, h, w = x.shape

        x = x.view(batch * frames, c, h, w)

        features = self.model(x)
        features = self.pool(features)

        features = features.view(batch, frames, 2048)

        # 🔥 LSTM
        lstm_out, _ = self.lstm(features)

        # take last frame output
        out = lstm_out[:, -1, :]

        out = self.linear1(out)

        return out