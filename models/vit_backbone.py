"""
Vision Transformer Backbone for Medical Imaging.

Lightweight ViT designed for federated learning:
- Small parameter count (~11M) to minimize communication cost
- Pre-LN for training stability
- CLS token output for downstream tasks
- Compatible with existing DetectionHead and ClassificationHead

Reference:
  UltraFedFM (Jiang et al., npj Digital Medicine 2025)
  Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PatchEmbedding(nn.Module):
    """
    Convert image into patch embeddings.

    Uses a Conv2d with kernel_size=stride=patch_size to efficiently
    split and embed patches in a single operation.

    Args:
        img_size: Input image size (assumed square)
        patch_size: Patch size (assumed square)
        in_chans: Number of input channels (3 for RGB, 1 for grayscale)
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for medical imaging.

    Lightweight configuration optimized for federated learning:
    - Fewer parameters → lower communication cost per round
    - Pre-LN (LayerNorm before attention) for stable training
    - CLS token aggregation for classification/detection heads

    Config presets:
      - "tiny":   embed_dim=192, depth=4,  heads=3  → ~3M params
      - "small":  embed_dim=384, depth=6,  heads=6  → ~11M params
      - "base":   embed_dim=768, depth=12, heads=12 → ~86M params

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Input channels
        embed_dim: Transformer embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio
        dropout: Dropout rate
        out_dim: Output projection dimension (for downstream heads)
        use_cls_token: Whether to use CLS token (True) or global avg pool
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        out_dim: int = 256,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_cls_token = use_cls_token

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token and positional embedding
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + (1 if use_cls_token else 0), embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks (Pre-LN)
        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]  # stochastic depth
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dpr[i],
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN for stability
            )
            for i in range(depth)
        ])

        # Final LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.proj = nn.Linear(embed_dim, out_dim)
        # Zero-init projection (doesn't affect pretrained encoder)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal."""
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.trunc_normal_(block.linear1.weight, std=0.02)
            nn.init.trunc_normal_(block.linear2.weight, std=0.02)
            nn.init.trunc_normal_(block.self_attn.in_proj_weight, std=0.02)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            nn.init.zeros_(block.linear1.bias)
            nn.init.zeros_(block.linear2.bias)

    @classmethod
    def from_config(cls, config: str = "small", **kwargs) -> "ViTBackbone":
        """Build from preset config."""
        presets = {
            "tiny": dict(embed_dim=192, depth=4, num_heads=3, out_dim=128),
            "small": dict(embed_dim=384, depth=6, num_heads=6, out_dim=256),
            "base": dict(embed_dim=768, depth=12, num_heads=12, out_dim=512),
        }
        if config not in presets:
            raise ValueError(f"Unknown config: {config}. Choose from {list(presets.keys())}")
        params = presets[config]
        params.update(kwargs)
        return cls(**params)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input images
            return_features: If True, return (output, pre-projection features)

        Returns:
            output: [B, out_dim] projected features
            features: [B, embed_dim] pre-projection features (if return_features)
        """
        B = x.shape[0]

        # Patch embed + pos embed
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Aggregate
        if self.cls_token is not None:
            features = x[:, 0]  # CLS token
        else:
            features = x[:, 1:].mean(dim=1)  # Global average pool (exclude CLS)

        # Project
        out = self.proj(features)

        if return_features:
            return out, features
        return out

    def get_encoder_for_mae(self) -> nn.Module:
        """
        Return the encoder part (without projection) for MAE pretraining.
        The projection layer is zero-initialized, so removing it doesn't
        affect the encoder's learned representations.
        """
        return nn.ModuleDict({
            "patch_embed": self.patch_embed,
            "cls_token": self.cls_token,
            "pos_embed": self.pos_embed,
            "blocks": self.blocks,
            "norm": self.norm,
        })

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        patch_params = sum(p.numel() for p in self.patch_embed.parameters())
        pos_params = self.pos_embed.numel()
        cls_params = self.cls_token.numel() if self.cls_token is not None else 0
        block_params = sum(p.numel() for p in self.blocks.parameters())
        proj_params = sum(p.numel() for p in self.proj.parameters())

        return {
            "total": total,
            "patch_embed": patch_params,
            "pos_embed": pos_params,
            "cls_token": cls_params,
            "blocks": block_params,
            "proj": proj_params,
        }


def build_backbone(config: str = "small", **kwargs) -> ViTBackbone:
    """Build a ViT backbone from config name."""
    return ViTBackbone.from_config(config, **kwargs)
