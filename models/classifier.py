"""
Classification and Detection heads for medical imaging tasks.

These heads sit on top of the ViT backbone and are trained locally
at each client (not shared in federated aggregation).

Design:
- ClassificationHead: For image-level tasks (disease classification, plane detection)
- DetectionHead: For object-level tasks (anomaly detection, measurement)
- Both support Shared Backbone + Local Head architecture for FL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassificationHead(nn.Module):
    """
    Classification head for medical image tasks.

    Architecture: FC -> BN -> GELU -> Dropout -> FC
    Supports multi-class and binary classification.

    Args:
        in_dim: Input feature dimension (must match backbone out_dim)
        n_classes: Number of output classes
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_dim: int = 256,
        n_classes: int = 6,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 2

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, in_dim] backbone output
        Returns:
            logits: [B, n_classes]
        """
        return self.head(features)


class MedicalClassifier(nn.Module):
    """
    Full medical image classifier: ViT Backbone + Classification Head.

    Federated learning design:
    - Backbone is shared across clients (aggregated on server)
    - Head is local to each client (not shared)
    - This is the "Shared Backbone + Local Head" architecture

    Args:
        backbone: ViTBackbone instance
        n_classes: Number of output classes
        head_hidden_dim: Hidden dim for classification head
        head_dropout: Dropout rate for classification head
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int = 6,
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(
            in_dim=backbone.out_dim,
            n_classes=n_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Args:
            x: [B, C, H, W] input images
            return_features: If True, also return backbone features

        Returns:
            logits: [B, n_classes]
            features: [B, out_dim] (if return_features)
        """
        features = self.backbone(x, return_features=False)
        logits = self.head(features)

        if return_features:
            return logits, features
        return logits

    def get_backbone_params(self) -> dict:
        """Get backbone parameters for FL aggregation."""
        return {k: v.clone() for k, v in self.backbone.state_dict().items()}

    def get_head_params(self) -> dict:
        """Get head parameters (local, not shared)."""
        return {k: v.clone() for k, v in self.head.state_dict().items()}

    def load_backbone_params(self, state_dict: dict):
        """Load aggregated backbone parameters from server."""
        self.backbone.load_state_dict(state_dict)

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": backbone_params + head_params,
            "comm_cost_MB": backbone_params * 4 / (1024 ** 2),  # float32
        }
