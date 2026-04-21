#!/usr/bin/env python3
"""
Federated Training with Prototype-Aware Aggregation.

Supports:
- FedAvg, FedProx, Prototype-Aware aggregation
- Noisy label robustness via prototype contrastive learning
- Non-IID data partitioning
- Multiple backbone sizes (Tiny/Small/Base)

Usage:
    python train_federated.py --config configs/default.yaml --num_clients 5 --rounds 50
    python train_federated.py --backbone tiny --aggregation prototype_aware --noise 0.2
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import random
import numpy as np

from models import build_vit, build_head, PrototypeBank
from data import SyntheticMedicalDataset, get_train_transforms, get_val_transforms, create_federated_dataloaders
from fl import FLServer, FLClient, ClientConfig
from utils import Logger, MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--backbone", type=str, default=None, choices=["tiny", "small", "base"])
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--aggregation", type=str, default=None,
                        choices=["fedavg", "fedprox", "prototype_aware"])
    parser.add_argument("--noise", type=float, default=None, help="Noisy label ratio")
    parser.add_argument("--non_iid_alpha", type=float, default=None)
    parser.add_argument("--prototype", action="store_true", default=None)
    parser.add_argument("--no_prototype", action="store_true")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to MAE encoder checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="runs/federated")
    parser.add_argument("--save_path", type=str, default="checkpoints/federated_model.pth")
    return parser.parse_args()


def load_config(args):
    """Load and merge configuration."""
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Override with CLI args
    if args.backbone:
        config.setdefault("model", {})["backbone"] = args.backbone
    if args.num_clients:
        config.setdefault("fl", {})["num_clients"] = args.num_clients
    if args.rounds:
        config.setdefault("fl", {})["rounds"] = args.rounds
    if args.local_epochs:
        config.setdefault("fl", {})["local_epochs"] = args.local_epochs
    if args.lr:
        config.setdefault("fl", {})["learning_rate"] = args.lr
    if args.batch_size:
        config.setdefault("fl", {})["batch_size"] = args.batch_size
    if args.aggregation:
        config.setdefault("fl", {})["aggregation"] = args.aggregation
    if args.noise is not None:
        config.setdefault("data", {})["noise_ratio"] = args.noise
    if args.non_iid_alpha is not None:
        config.setdefault("data", {})["non_iid_alpha"] = args.non_iid_alpha
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed
    if args.prototype:
        config.setdefault("prototype", {})["enabled"] = True
    if args.no_prototype:
        config.setdefault("prototype", {})["enabled"] = False

    return config


def build_model(config, device, pretrained_path=None):
    """Build the full model (backbone + head)."""
    model_cfg = config.get("model", {})
    backbone_size = model_cfg.get("backbone", "tiny")
    img_size = model_cfg.get("img_size", 224)
    num_classes = config.get("data", {}).get("num_classes", 6)
    head_type = model_cfg.get("head", "classification")
    hidden_dim = model_cfg.get("hidden_dim", 256)

    # Build backbone
    backbone = build_vit(backbone_size, img_size=img_size)

    # Load pretrained encoder if provided
    if pretrained_path:
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        backbone.load_state_dict(ckpt["encoder_state_dict"])
        print(f"[INFO] Loaded pretrained encoder from {pretrained_path}")

    # Get embed dim from backbone
    embed_dim = backbone.embed_dim

    # Build head
    head = build_head(head_type, in_dim=embed_dim, num_classes=num_classes, hidden_dim=hidden_dim)

    # Combine
    class MedicalModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

        def get_backbone_params(self):
            return {k: v for k, v in self.backbone.named_parameters()}

        def get_head_params(self):
            return {k: v for k, v in self.head.named_parameters()}

        def get_num_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    model = MedicalModel(backbone, head).to(device)
    return model, embed_dim


def main():
    args = parse_args()
    config = load_config(args)

    # Seed
    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Config extraction
    model_cfg = config.get("model", {})
    fl_cfg = config.get("fl", {})
    data_cfg = config.get("data", {})
    proto_cfg = config.get("prototype", {})
    train_cfg = config.get("training", {})

    backbone_size = model_cfg.get("backbone", "tiny")
    img_size = model_cfg.get("img_size", 224)
    num_clients = fl_cfg.get("num_clients", 5)
    num_rounds = fl_cfg.get("rounds", 50)
    local_epochs = fl_cfg.get("local_epochs", 3)
    batch_size = fl_cfg.get("batch_size", 32)
    lr = fl_cfg.get("learning_rate", 1e-3)
    wd = train_cfg.get("finetune_wd", 5e-5)
    aggregation = fl_cfg.get("aggregation", "prototype_aware")
    noise_ratio = data_cfg.get("noise_ratio", 0.2)
    non_iid_alpha = data_cfg.get("non_iid_alpha", 0.5)
    num_samples = data_cfg.get("num_samples_per_client", 500)
    num_classes = data_cfg.get("num_classes", 6)
    use_prototype = proto_cfg.get("enabled", True)

    # Logger
    logger = Logger(
        name=f"fl_{backbone_size}_{aggregation}",
        log_dir=args.log_dir,
        use_tensorboard=config.get("logging", {}).get("use_tensorboard", False),
    )

    logger.log(f"Device: {device}")
    logger.log(f"Backbone: ViT-{backbone_size}")
    logger.log(f"Clients: {num_clients} | Rounds: {num_rounds} | Local epochs: {local_epochs}")
    logger.log(f"Aggregation: {aggregation}")
    logger.log(f"Noise ratio: {noise_ratio} | Non-IID alpha: {non_iid_alpha}")
    logger.log(f"Prototype contrastive: {use_prototype}")

    # Build model
    logger.log("Building model...")
    model, embed_dim = build_model(config, device, args.pretrained)
    logger.log(f"Model params: {model.get_num_params():,}")

    # Build prototype bank
    prototype_bank = None
    if use_prototype:
        prototype_bank = PrototypeBank(
            n_classes=num_classes,
            embed_dim=embed_dim,
            temperature=proto_cfg.get("temperature", 0.07),
            momentum=proto_cfg.get("momentum", 0.9),
            confidence_threshold=proto_cfg.get("confidence_threshold", 0.5),
        ).to(device)
        logger.log(f"Prototype bank: {num_classes} classes, dim={embed_dim}")

    # Data
    logger.log("Creating federated data splits...")
    full_dataset = SyntheticMedicalDataset(
        num_samples=num_samples * num_clients,
        num_classes=num_classes,
        img_size=img_size,
        noise_ratio=noise_ratio,
        transform=get_train_transforms(img_size=img_size),
    )

    # Validation set (clean labels)
    val_dataset = SyntheticMedicalDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        img_size=img_size,
        noise_ratio=0.0,
        transform=get_val_transforms(img_size=img_size),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create federated data loaders (Non-IID)
    client_loaders = create_federated_dataloaders(
        full_dataset, num_clients, batch_size=batch_size,
        alpha=non_iid_alpha, seed=seed,
    )

    for i, loader in enumerate(client_loaders):
        logger.log(f"  Client {i}: {len(loader.dataset)} samples, {len(loader)} batches")

    # Build FL server
    logger.log("Initializing FL server...")
    server = FLServer(
        model=model,
        aggregation=aggregation,
        prototype_bank=prototype_bank,
    )

    # Build FL clients
    logger.log("Initializing FL clients...")
    clients = []
    for i in range(num_clients):
        client_model = build_model(config, device, args.pretrained)[0]
        client = FLClient(
            model=client_model,
            config=ClientConfig(
                client_id=f"client_{i}",
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=lr,
                weight_decay=wd,
                device=str(device),
            ),
            train_loader=client_loaders[i],
            prototype_bank=PrototypeBank(
                n_classes=num_classes, embed_dim=embed_dim,
                temperature=proto_cfg.get("temperature", 0.07),
            ).to(device) if use_prototype else None,
        )
        clients.append(client)
        logger.log(f"  Client {i}: {client_model.get_num_params():,} params")

    # Evaluation function
    def eval_fn(model):
        model.eval()
        tracker = MetricsTracker(num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                tracker.update(preds.cpu(), labels.cpu(), loss.item())
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        metrics = tracker.compute()
        metrics["loss"] = total_loss / max(total_samples, 1)
        return metrics["loss"], metrics["accuracy"]

    # Federated training loop
    logger.log("=" * 60)
    logger.log("Starting federated training...")
    logger.log("=" * 60)

    min_clients = fl_cfg.get("min_clients_per_round", max(1, num_clients // 2))

    for round_num in range(1, num_rounds + 1):
        # Select participating clients
        num_participants = max(min_clients, random.randint(min_clients, num_clients))
        participants = random.sample(clients, num_participants)

        logger.log(f"--- Round {round_num}/{num_rounds} ({num_participants} clients) ---")

        # Distribute global model
        updates = []
        for client in participants:
            client.receive_global_model(server.get_global_params())

            # Train locally
            update = client.train_round()
            updates.append(update)

            logger.log(
                f"  {client.config.client_id}: "
                f"loss={update.local_loss:.4f}, "
                f"acc={update.local_accuracy:.4f}, "
                f"samples={update.num_samples}, "
                f"time={update.training_time:.1f}s"
            )

        # Aggregate
        result = server.aggregate_round(updates)

        # Evaluate
        if round_num % config.get("logging", {}).get("eval_interval", 5) == 0:
            loss, acc = server.evaluate_global_model(eval_fn)
            result.global_loss = loss
            result.global_accuracy = acc

        logger.log_round(round_num, {
            "loss": result.global_loss,
            "accuracy": result.global_accuracy,
            "num_clients": result.num_participants,
            "comm_cost_mb": result.communication_cost_mb,
        })

        # Save checkpoint
        if round_num % config.get("logging", {}).get("save_interval", 10) == 0:
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": server.model.state_dict(),
                "round": round_num,
                "accuracy": result.global_accuracy,
                "config": config,
            }, save_path)
            logger.log(f"Checkpoint saved: {save_path}")

    # Final evaluation
    logger.log("=" * 60)
    final_loss, final_acc = server.evaluate_global_model(eval_fn)
    logger.log(f"Training complete!")
    logger.log(f"Final loss: {final_loss:.4f} | Final accuracy: {final_acc:.4f}")

    # Save final model
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": server.model.state_dict(),
        "round": num_rounds,
        "accuracy": final_acc,
        "config": config,
    }, save_path)
    logger.log(f"Model saved to: {save_path}", level="SUCCESS")

    # Print summary
    summary = server.get_summary()
    logger.log(f"\nTraining summary:")
    logger.log(f"  Total rounds: {summary['current_round']}")
    logger.log(f"  Model params: {summary['model_params']:,}")
    logger.log(f"  Aggregation: {summary['aggregation']}")

    logger.close()


if __name__ == "__main__":
    main()
