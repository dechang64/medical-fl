#!/usr/bin/env python3
"""Evaluate a trained model."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from models.vit import build_vit
from models.heads import build_head
from data.dataset import SyntheticMedicalDataset
from utils.metrics import MetricsTracker, compute_confusion_matrix, format_metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="tiny")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bb = build_vit(args.backbone, img_size=args.img_size)
    model = nn.Sequential(bb, build_head("classification", in_dim=bb.out_dim, num_classes=args.num_classes))
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    ds = SyntheticMedicalDataset(num_samples=args.num_samples, num_classes=args.num_classes, img_size=args.img_size, noise_ratio=0.0, seed=99, normalize=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    tracker = MetricsTracker(args.num_classes)
    with torch.no_grad():
        for batch in loader:
            tracker.update(model(batch["image"]), batch["label"])

    metrics = tracker.compute()
    cm = compute_confusion_matrix(tracker.all_preds, tracker.all_targets, args.num_classes)
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(format_metrics(metrics))
    print(f"\nConfusion Matrix:\n{cm}")
    print("=" * 50)

if __name__ == "__main__":
    main()
