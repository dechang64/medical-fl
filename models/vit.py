"""ViT backbone. Tiny/Small/Base."""
import torch, torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class ViTConfig:
    img_size: int = 64; patch_size: int = 8; in_channels: int = 3
    embed_dim: int = 192; depth: int = 6; num_heads: int = 3
    mlp_ratio: float = 4.0; dropout: float = 0.1
    @property
    def num_patches(self): return (self.img_size // self.patch_size) ** 2
    @property
    def mlp_dim(self): return int(self.embed_dim * self.mlp_ratio)
    @classmethod
    def from_name(cls, name, **kw):
        p = {"tiny": dict(embed_dim=192,depth=6,num_heads=3),
             "small": dict(embed_dim=384,depth=12,num_heads=6),
             "base": dict(embed_dim=768,depth=12,num_heads=12)}
        return cls(**{**p[name], **kw})

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp_dim, dim), nn.Dropout(drop))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))

class ViTBackbone(nn.Module):
    def __init__(self, cfg: ViTConfig, out_dim: Optional[int] = None):
        super().__init__()
        self.config = cfg
        self.patch_embed = PatchEmbedding(cfg.img_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches + 1, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_dim, cfg.dropout) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.out_dim = out_dim or cfg.embed_dim
        self.head = nn.Linear(cfg.embed_dim, self.out_dim) if out_dim else nn.Identity()
        self._init_weights()
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02); nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        B = x.shape[0]; x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1); x = torch.cat([cls, x], 1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x[:, 0]))
    def forward_features(self, x):
        B = x.shape[0]; x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1); x = torch.cat([cls, x], 1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks: x = blk(x)
        return self.norm(x[:, 0])
    def get_num_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_vit(size="tiny", img_size=64, out_dim=None, **kw):
    return ViTBackbone(ViTConfig.from_name(size, img_size=img_size, **kw), out_dim)

VIT_CONFIGS = {"tiny": dict(embed_dim=192,depth=6,num_heads=3), "small": dict(embed_dim=384,depth=12,num_heads=6), "base": dict(embed_dim=768,depth=12,num_heads=12)}
