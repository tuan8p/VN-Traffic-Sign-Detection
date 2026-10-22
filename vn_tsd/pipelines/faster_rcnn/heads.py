"""Custom RoI predictor; the pretrained TwoMLPHead before it stays intact.

Preserves TorchVision's (class_logits, box_regression) interface. The shared
projection is new; changing the class count alone is the standard baseline.
"""
from __future__ import annotations

from torch import nn


class CustomFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)
        nn.init.kaiming_normal_(self.projection[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.projection[0].bias)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.zeros_(self.bbox_pred.bias)

    def forward(self, x):
        # TorchVision predictors accept [N,C] or pooled [N,C,1,1], including N=0.
        if x.dim() == 4 and tuple(x.shape[-2:]) == (1, 1):
            x = x.flatten(start_dim=1)
        if x.dim() != 2:
            raise ValueError("RoI predictor expects [N,C] or [N,C,1,1] features")
        features = self.projection(x)
        return self.cls_score(features), self.bbox_pred(features)
