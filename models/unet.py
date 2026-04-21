"""Lightweight UNet decoder for segmentation."""
import torch, torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())
    def forward(self, x): return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(gate_ch, skip_ch // 2, 1), nn.BatchNorm2d(skip_ch // 2), nn.GELU(),
            nn.Conv2d(skip_ch // 2, skip_ch, 1), nn.BatchNorm2d(skip_ch), nn.Sigmoid())
    def forward(self, gate, skip): return skip * self.gate(gate)

class UNetDecoder(nn.Module):
    def __init__(self, encoder_dim=192, num_classes=6, use_attention=False):
        super().__init__()
        channels = [256, 128, 64, 32]
        self.ups = nn.ModuleList([nn.ConvTranspose2d(encoder_dim if i == 0 else channels[i-1], channels[i], 2, stride=2) for i in range(4)])
        self.convs = nn.ModuleList([DoubleConv(channels[i] * 2, channels[i]) for i in range(4)])
        self.attns = nn.ModuleList([AttentionGate(channels[i], channels[i]) for i in range(4)]) if use_attention else None
        self.final = nn.Conv2d(channels[-1], num_classes, 1)
    def forward(self, features, skips=None):
        x = features
        for i, (up, conv) in enumerate(zip(self.ups, self.convs)):
            x = up(x)
            if skips and i < len(skips):
                s = self.attns[i](x, skips[i]) if self.attns else skips[i]
                x = torch.cat([x, s], 1)
            x = conv(x)
        return self.final(x)
