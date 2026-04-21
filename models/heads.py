"""Task heads: classification, segmentation, detection."""
import torch, torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_dim=192, num_classes=6, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))
    def forward(self, x): return self.head(x)

class SegmentationHead(nn.Module):
    def __init__(self, in_dim=192, num_classes=6, hidden_dim=256):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, hidden_dim, 2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, num_classes, 1))
    def forward(self, features, H, W):
        x = self.up(features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H // 4, W // 4))
        return self.conv(x)

class DetectionHead(nn.Module):
    def __init__(self, in_dim=192, num_classes=6, num_queries=10, hidden_dim=128):
        super().__init__()
        self.query = nn.Embedding(num_queries, hidden_dim)
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(hidden_dim, 4, hidden_dim*4, batch_first=True), 2)
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)
        self.box_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4))
    def forward(self, features):
        B, N, D = features.shape
        memory = self.proj(features)
        q = self.query.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.dec(q, memory)
        return {"class_logits": self.cls_head(out), "boxes": self.box_head(out).sigmoid()}

def build_head(task, in_dim=192, num_classes=6, **kw):
    return {"classification": ClassificationHead, "segmentation": SegmentationHead, "detection": DetectionHead}[task](in_dim=in_dim, num_classes=num_classes, **kw)
