#!/usr/bin/env python3
"""MAE self-supervised pretraining."""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from models.mae import build_mae
from data.dataset import SyntheticMedicalDataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="tiny")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save", default="checkpoints/mae.pth")
    args = p.parse_args()

    ds = SyntheticMedicalDataset(num_samples=1000, num_classes=6, img_size=args.img_size, seed=42, normalize=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    mae = build_mae(args.backbone, img_size=args.img_size)
    opt = torch.optim.AdamW(mae.parameters(), lr=args.lr, weight_decay=5e-5)

    print(f"MAE pretraining: {args.backbone}, {args.epochs} epochs")
    print(f"Encoder params: {mae.get_num_params(encoder_only=True):,}")
    print(f"Total params: {mae.get_num_params():,}")

    for epoch in range(1, args.epochs + 1):
        mae.train(); total_loss = 0.0; n = 0
        for batch in loader:
            loss, _, _ = mae(batch["image"])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * batch["image"].size(0); n += batch["image"].size(0)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss/n:.4f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({"model_state_dict": mae.state_dict(), "backbone": args.backbone}, args.save)
    print(f"Saved to {args.save}")

if __name__ == "__main__":
    main()
