#!/usr/bin/env python3
"""
Self-Supervised Pretraining with Masked Autoencoder (MAE).

Pretrains a ViT backbone on local data without any labels.
The encoder can then be used for downstream federated fine-tuning.

Usage:
    python train_pretrain.py --config configs/default.yaml --mode mae
    python train_pretrain.py --backbone tiny --epochs 100 --data synthetic
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from models import build_vit, build_mae
from data import SyntheticMedicalDataset, get_train_transforms, get_val_transforms
from utils import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="MAE Self-Supervised Pretraining")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--backbone", type=str, default=None, choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_dir", type=str, default="runs/mae_pretrain")
    parser.add_argument("--save_path", type=str, default="checkpoints/mae_encoder.pth")
    return parser.parse_args()


def load_config(args):
    """Load and merge configuration."""
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Override with command line args
    if args.backbone:
        config.setdefault("model", {})["backbone"] = args.backbone
    if args.epochs:
        config.setdefault("training", {})["pretrain_epochs"] = args.epochs
    if args.lr:
        config.setdefault("training", {})["pretrain_lr"] = args.lr
    if args.batch_size:
        config.setdefault("fl", {})["batch_size"] = args.batch_size
    if args.num_samples:
        config.setdefault("data", {})["num_samples_per_client"] = args.num_samples
    if args.img_size:
        config.setdefault("model", {})["img_size"] = args.img_size
    if args.mask_ratio:
        config.setdefault("mae", {})["mask_ratio"] = args.mask_ratio

    return config


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Create cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, device, logger, epoch, log_interval=10):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    num_batches = 0

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)

        loss, _, _ = model(images)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_samples
            lr = optimizer.param_groups[0]["lr"]
            logger.log(
                f"Epoch {epoch} [{batch_idx+1}/{len(loader)}] | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
            )

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, device):
    """Validate reconstruction loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for images, _ in loader:
        images = images.to(device)
        loss, _, _ = model(images)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / max(total_samples, 1)


def main():
    args = parse_args()
    config = load_config(args)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Model config
    model_cfg = config.get("model", {})
    backbone_size = model_cfg.get("backbone", "tiny")
    img_size = model_cfg.get("img_size", 224)

    # MAE config
    mae_cfg = config.get("mae", {})
    mask_ratio = mae_cfg.get("mask_ratio", 0.75)
    decoder_dim = mae_cfg.get("decoder_dim", 128)
    decoder_depth = mae_cfg.get("decoder_depth", 4)
    decoder_heads = mae_cfg.get("decoder_heads", 4)

    # Training config
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("pretrain_epochs", 100)
    lr = train_cfg.get("pretrain_lr", 1.5e-4)
    wd = train_cfg.get("pretrain_wd", 5e-2)
    warmup = train_cfg.get("pretrain_warmup", 10)

    # Data config
    data_cfg = config.get("data", {})
    num_samples = data_cfg.get("num_samples_per_client", 2000)
    batch_size = config.get("fl", {}).get("batch_size", 64)

    # Logger
    logger = Logger(
        name=f"mae_{backbone_size}",
        log_dir=args.log_dir,
        use_tensorboard=config.get("logging", {}).get("use_tensorboard", False),
    )

    logger.log(f"Device: {device}")
    logger.log(f"Backbone: ViT-{backbone_size}")
    logger.log(f"Image size: {img_size}")
    logger.log(f"Mask ratio: {mask_ratio}")
    logger.log(f"Epochs: {epochs}, LR: {lr}, WD: {wd}")

    # Build model
    logger.log("Building MAE model...")
    mae = build_mae(
        backbone_size=backbone_size,
        img_size=img_size,
        mask_ratio=mask_ratio,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
    ).to(device)

    total_params = mae.get_num_params()
    encoder_params = mae.get_num_params(encoder_only=True)
    logger.log(f"Total params: {total_params:,} | Encoder params: {encoder_params:,}")

    # Data
    logger.log("Loading synthetic medical data...")
    train_dataset = SyntheticMedicalDataset(
        num_samples=num_samples,
        num_classes=data_cfg.get("num_classes", 6),
        img_size=img_size,
        transform=get_train_transforms(img_size=img_size),
    )
    val_dataset = SyntheticMedicalDataset(
        num_samples=num_samples // 5,
        num_classes=data_cfg.get("num_classes", 6),
        img_size=img_size,
        transform=get_val_transforms(img_size=img_size),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.log(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, warmup, total_steps)

    # Training loop
    logger.log("=" * 60)
    logger.log("Starting MAE pretraining...")
    logger.log("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            mae, train_loader, optimizer, scheduler, device, logger, epoch,
            log_interval=config.get("logging", {}).get("log_interval", 10),
        )

        val_loss = validate(mae, val_loader, device)

        logger.log_epoch(epoch, {"loss": train_loss, "val_loss": val_loss}, phase="pretrain")

        # Save best encoder
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "encoder_state_dict": mae.get_encoder().state_dict(),
                "config": {
                    "backbone_size": backbone_size,
                    "img_size": img_size,
                    "mask_ratio": mask_ratio,
                },
                "epoch": epoch,
                "val_loss": val_loss,
            }, save_path)
            logger.log(f"Saved best encoder (val_loss: {val_loss:.4f})", level="SUCCESS")

    logger.log("=" * 60)
    logger.log(f"Pretraining complete! Best val_loss: {best_val_loss:.4f}")
    logger.log(f"Encoder saved to: {args.save_path}")
    logger.log("=" * 60)

    logger.close()


if __name__ == "__main__":
    main()
