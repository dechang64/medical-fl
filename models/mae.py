"""Masked Autoencoder for federated self-supervised pretraining."""
import torch, torch.nn as nn, torch.nn.functional as F
from .vit import ViTBackbone, ViTConfig

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim, dec_dim, dec_depth, dec_heads, num_patches, patch_size, in_ch=3):
        super().__init__()
        self.dec_embed = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, num_patches, dec_dim))
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dec_dim, dec_heads, dec_dim*4, batch_first=True, dropout=0.1) for _ in range(dec_depth)])
        self.norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, patch_size * patch_size * in_ch)
        self.patch_size = patch_size; self.in_ch = in_ch
    def forward(self, x, ids_restore):
        x = self.dec_embed(x)
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] - x.shape[1], -1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x_ = x_ + self.dec_pos
        for blk in self.blocks: x_ = blk(x_)
        return self.pred(self.norm(x_))

class MaskedAutoencoder(nn.Module):
    def __init__(self, backbone, mask_ratio=0.75, dec_dim=128, dec_depth=4, dec_heads=4):
        super().__init__()
        self.encoder = backbone
        cfg = backbone.config
        self.decoder = MAEDecoder(cfg.embed_dim, dec_dim, dec_depth, dec_heads, cfg.num_patches, cfg.patch_size, cfg.in_channels)
        self.mask_ratio = mask_ratio
        self.patch_size = cfg.patch_size
        self.num_patches = cfg.num_patches
    def random_mask(self, x):
        B, N, D = x.shape; len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore
    def forward(self, imgs):
        x = self.encoder.patch_embed(imgs)
        B = x.shape[0]
        cls = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], 1)
        x = x + self.encoder.pos_embed
        x_masked, mask, ids_restore = self.random_mask(x[:, 1:])
        x_masked = torch.cat([cls, x_masked], 1)
        for blk in self.encoder.blocks: x_masked = blk(x_masked)
        x_masked = self.encoder.norm(x_masked[:, 1:])
        pred = self.decoder(x_masked, ids_restore)
        H = W = int(self.num_patches ** 0.5)
        target = self.encoder.patch_embed.proj(imgs).flatten(2).transpose(1, 2)
        loss = F.mse_loss(pred, target)
        return loss, mask, pred
    def get_encoder(self): return self.encoder
    def get_num_params(self, encoder_only=False):
        return self.encoder.get_num_params() if encoder_only else sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_mae(backbone_size="tiny", img_size=64, mask_ratio=0.75, **kw):
    from .vit import build_vit
    return MaskedAutoencoder(build_vit(backbone_size, img_size=img_size, **kw), mask_ratio)
