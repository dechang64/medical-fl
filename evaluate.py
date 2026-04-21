#!/usr/bin/env python3
"""
Model Evaluation Script.

Evaluates a trained model on a test set with comprehensive metrics.

Usage:
    python evaluate.py --checkpoint checkpoints/federated_model.pth --config configs/default.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from models import build_vit, build_head
from data import SyntheticMedicalDataset, get_val_transforms
from utils import Logger, MetricsTracker, format_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, default="synthetic")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_config(args):
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def build_model_from_checkpoint(checkpoint_path, config, device):
    """Build model and load checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = config.get("model", {})
    backbone_size = model_cfg.get("backbone", "tiny")
    img_size = model_cfg.get("img_size", 224)
    num_classes = config.get("data", {}).get("num_classes", 6)
    head_type = model_cfg.get("head", "classification")
    hidden_dim = model_cfg.get("hidden_dim", 256)

    backbone = build_vit(backbone_size, img_size=img_size)
    head = build_head(head_type, in_dim=backbone.embed_dim, num_classes=num_classes, hidden_dim=hidden_dim)

    class MedicalModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x))

    model = MedicalModel(backbone, head).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    return model, ckpt


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Run full evaluation."""
    model.eval()
    tracker = MetricsTracker(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        tracker.update(preds.cpu(), labels.cpu(), loss.item(), probs.cpu())

    return tracker.compute()


def main():
    args = parse_args()
    config = load_config(args)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger = Logger(name="evaluation", log_dir="runs/eval")

    # Build model
    logger.log(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = build_model_from_checkpoint(args.checkpoint, config, device)
    logger.log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    if "round" in ckpt:
        logger.log(f"Trained for {ckpt['round']} rounds")
    if "accuracy" in ckpt:
        logger.log(f"Training accuracy: {ckpt['accuracy']:.4f}")

    # Data
    img_size = config.get("model", {}).get("img_size", 224)
    num_classes = config.get("data", {}).get("num_classes", 6)

    test_dataset = SyntheticMedicalDataset(
        num_samples=args.num_samples,
        num_classes=num_classes,
        img_size=img_size,
        noise_ratio=0.0,  # Clean labels for evaluation
        transform=get_val_transforms(img_size=img_size),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.log(f"Test samples: {len(test_dataset)}")

    # Evaluate
    logger.log("Running evaluation...")
    metrics = evaluate(model, test_loader, device, num_classes)

    # Print results
    logger.log("=" * 60)
    logger.log("EVALUATION RESULTS")
    logger.log("=" * 60)
    logger.log(format_metrics(metrics))
    logger.log("=" * 60)

    logger.close()


if __name__ == "__main__":
    main()
