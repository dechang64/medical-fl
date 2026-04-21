"""
Detection head for medical imaging (retained from embodied-fl).

Lightweight single-stage detector adapted for medical images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm2d + SiLU activation."""
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))


class DetectionHead(nn.Module):
    """
    Detection head for medical image object detection.
    
    Anchor-free, center-based detection.
    """
    def __init__(self, in_dim=256, n_classes=6, max_objects=10):
        super().__init__()
        self.n_classes = n_classes
        self.max_objects = max_objects

        self.box_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, max_objects * 4),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, max_objects * n_classes),
        )
        self.obj_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, max_objects),
        )

    def forward(self, features):
        B = features.shape[0]
        boxes = self.box_head(features).view(B, self.max_objects, 4)
        cls = self.cls_head(features).view(B, self.max_objects, self.n_classes)
        obj = self.obj_head(features).view(B, self.max_objects)
        return boxes, cls, obj


class MedicalDetector(nn.Module):
    """Full detector: ViT Backbone + Detection Head."""
    def __init__(self, backbone, n_classes=6, max_objects=10):
        super().__init__()
        self.backbone = backbone
        self.head = DetectionHead(
            in_dim=backbone.out_dim,
            n_classes=n_classes,
            max_objects=max_objects,
        )

    def forward(self, x, return_features=False):
        features = self.backbone(x, return_features=False)
        boxes, cls, obj = self.head(features)
        if return_features:
            return boxes, cls, obj, features
        return boxes, cls, obj

    def get_backbone_params(self):
        return self.backbone.state_dict()

    def get_head_params(self):
        return self.head.state_dict()

    def load_backbone_params(self, state_dict):
        self.backbone.load_state_dict(state_dict)
