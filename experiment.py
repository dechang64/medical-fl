"""
Medical-FL: Federated Learning Framework for Medical Imaging

A complete framework for privacy-preserving medical image analysis
using federated learning, featuring:

1. ViT Backbone (lightweight, ~11M params)
2. MAE Self-Supervised Pretraining (no labels needed)
3. Prototype Contrastive Learning (robust to noisy labels)
4. Multiple aggregation strategies (FedAvg, volume-weighted, prototype-aware)
5. Blockchain audit chain integration (via Rust gRPC)

Usage:
    # Quick test (CPU, ~2 min)
    python experiment.py --mode quick

    # Paper-quality experiment (GPU recommended, ~30 min)
    python experiment.py --mode paper

    # Full ablation study (GPU, ~2 hr)
    python experiment.py --mode full

    # MAE pretraining only
    python experiment.py --mode pretrain

    # Compare aggregation strategies
    python experiment.py --mode compare
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    ViTBackbone,
    MAEPretrainer,
    PrototypeBank,
    ClassificationHead,
    MedicalClassifier,
)
from data import (
    SyntheticFetalUltrasound,
    get_federated_dataloaders,
    print_data_stats,
    PLANE_NAMES,
    N_CLASSES,
)
from utils import (
    fedavg_aggregate,
    volume_weighted_aggregate,
    prototype_aware_aggregate,
    get_model_params,
    set_model_params,
    count_parameters,
    communication_cost,
    evaluate_classifier,
    format_results,
)


# ═══════════════════════════════════════════════════════════════
#  Experiment Configurations
# ═══════════════════════════════════════════════════════════════

CONFIGS = {
    "quick": {
        "description": "Quick test (CPU, ~2 min)",
        "n_clients": 3,
        "samples_per_client": 100,
        "img_size": 64,
        "batch_size": 16,
        "fl_rounds": 4,
        "local_epochs": 1,
        "lr": 1e-3,
        "noise_ratio": 0.2,
        "noniid_alpha": 0.5,
        "use_mae": False,
        "mae_epochs": 0,
        "use_prototype": True,
        "aggregation": "fedavg",
        "backbone_config": "tiny",
        "seed": 42,
    },
    "paper": {
        "description": "Paper-quality experiment (GPU, ~30 min)",
        "n_clients": 3,
        "samples_per_client": 200,
        "img_size": 64,
        "batch_size": 32,
        "fl_rounds": 10,
        "local_epochs": 1,
        "lr": 5e-4,
        "noise_ratio": 0.2,
        "noniid_alpha": 0.3,
        "use_mae": True,
        "mae_epochs": 2,
        "use_prototype": True,
        "aggregation": "volume_weighted",
        "backbone_config": "tiny",
        "seed": 42,
    },
    "full": {
        "description": "Full ablation study (GPU, ~2 hr)",
        "n_clients": 5,
        "samples_per_client": 600,
        "img_size": 128,
        "batch_size": 32,
        "fl_rounds": 30,
        "local_epochs": 3,
        "lr": 3e-4,
        "noise_ratio": 0.3,
        "noniid_alpha": 0.2,
        "use_mae": True,
        "mae_epochs": 20,
        "use_prototype": True,
        "aggregation": "prototype_aware",
        "backbone_config": "small",
        "seed": 42,
    },
    "pretrain": {
        "description": "MAE pretraining only",
        "n_clients": 3,
        "samples_per_client": 200,
        "img_size": 64,
        "batch_size": 16,
        "fl_rounds": 5,
        "local_epochs": 1,
        "lr": 1e-3,
        "noise_ratio": 0.0,
        "noniid_alpha": 0.5,
        "use_mae": True,
        "mae_epochs": 10,
        "use_prototype": False,
        "aggregation": "fedavg",
        "backbone_config": "tiny",
        "seed": 42,
    },
    "compare": {
        "description": "Compare aggregation strategies",
        "n_clients": 5,
        "samples_per_client": 300,
        "img_size": 128,
        "batch_size": 32,
        "fl_rounds": 15,
        "local_epochs": 2,
        "lr": 5e-4,
        "noise_ratio": 0.2,
        "noniid_alpha": 0.3,
        "use_mae": True,
        "mae_epochs": 5,
        "use_prototype": True,
        "aggregation": "all",
        "backbone_config": "small",
        "seed": 42,
    },
}


# ═══════════════════════════════════════════════════════════════
#  MAE Pretraining
# ═══════════════════════════════════════════════════════════════

def run_mae_pretraining(config: dict) -> dict:
    """Run federated MAE pretraining."""
    print("\n" + "=" * 60)
    print("  Phase 1: MAE Self-Supervised Pretraining")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Data (no labels needed for MAE)
    client_loaders, _ = get_federated_dataloaders(
        n_clients=config["n_clients"],
        samples_per_client=config["samples_per_client"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        noise_ratio=0.0,  # No labels needed
        noniid_alpha=config["noniid_alpha"],
        seed=config["seed"],
    )

    # MAE model
    encoder_backbone = ViTBackbone.from_config(
        config["backbone_config"],
        img_size=config["img_size"],
        patch_size=8 if config["img_size"] <= 64 else 16,
        in_chans=3,
        embed_dim=192 if config["backbone_config"] == "tiny" else 384,
    ).to(device)
    mae = MAEPretrainer(
        backbone=encoder_backbone,
        mask_ratio=0.75,
        decoder_dim=128 if config["backbone_config"] == "tiny" else 256,
    ).to(device)

    n_params = count_parameters(mae)
    comm = communication_cost(get_model_params(mae))
    print(f"  MAE parameters: {n_params:,}")
    print(f"  Communication cost per round: {comm['MB']:.2f} MB")

    # Optimizer
    optimizer = torch.optim.AdamW(mae.parameters(), lr=config["lr"], weight_decay=0.05)

    # Training loop
    history = []
    for epoch in range(config["mae_epochs"]):
        epoch_loss = 0
        n_batches = 0

        for client_id, loader in enumerate(client_loaders):
            mae.train()
            for batch in loader:
                images = batch[0].to(device)

                loss, metrics = mae(images)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss})

        if (epoch + 1) % max(1, config["mae_epochs"] // 5) == 0:
            print(f"    Epoch {epoch+1}/{config['mae_epochs']}  Loss: {avg_loss:.4f}")

    # Extract encoder for downstream use
    encoder_state = mae.get_encoder_state_dict()
    print(f"  Pretraining complete. Final loss: {history[-1]['loss']:.4f}")

    return {
        "global_encoder": encoder_state,
        "history": history,
        "n_params": n_params,
    }


# ═══════════════════════════════════════════════════════════════
#  Federated Training
# ═══════════════════════════════════════════════════════════════

def run_federated_training(
    config: dict,
    pretrained_encoder: dict = None,
    aggregation: str = "fedavg",
) -> dict:
    """Run federated training with specified aggregation."""
    print(f"\n  Aggregation: {aggregation}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    client_loaders, test_loader = get_federated_dataloaders(
        n_clients=config["n_clients"],
        samples_per_client=config["samples_per_client"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        noise_ratio=config["noise_ratio"],
        noniid_alpha=config["noniid_alpha"],
        seed=config["seed"],
    )

    # Global model
    patch_size = 8 if config["img_size"] <= 64 else 16
    backbone = ViTBackbone.from_config(
        config["backbone_config"],
        img_size=config["img_size"],
        patch_size=patch_size,
        in_chans=3,
    )

    if pretrained_encoder is not None:
        backbone.patch_embed.load_state_dict(pretrained_encoder["patch_embed"])
        backbone.cls_token.data.copy_(pretrained_encoder["cls_token"])
        backbone.pos_embed.data.copy_(pretrained_encoder["pos_embed"])
        backbone.blocks.load_state_dict(pretrained_encoder["blocks"])
        backbone.norm.load_state_dict(pretrained_encoder["norm"])
        print("  Loaded pretrained encoder weights")

    classifier = MedicalClassifier(
        backbone=backbone,
        n_classes=N_CLASSES,
        head_dropout=0.3,
    )

    params_info = classifier.count_parameters()
    comm = communication_cost(classifier.get_backbone_params())
    print(f"  Backbone params: {params_info['backbone']:,}")
    print(f"  Head params: {params_info['head']:,}")
    print(f"  Comm cost per round: {params_info['comm_cost_MB']:.2f} MB")

    # Prototype bank (if enabled)
    prototype_bank = None
    if config["use_prototype"]:
        prototype_bank = PrototypeBank(
            n_classes=N_CLASSES,
            embed_dim=backbone.out_dim,
        ).to(device)

    # Global optimizer (for prototype updates)
    global_optimizer = None
    if prototype_bank is not None:
        global_optimizer = torch.optim.SGD(
            prototype_bank.parameters(), lr=0.01
        )

    # Training history
    history = []

    for round_num in range(config["fl_rounds"]):
        round_start = time.time()

        # Broadcast global model to clients
        global_backbone = classifier.get_backbone_params()

        # Client updates
        client_params_list = []
        client_losses = []
        client_sample_counts = []
        client_prototypes = []

        for client_id, loader in enumerate(client_loaders):
            # Local model
            local_classifier = MedicalClassifier(
                backbone=ViTBackbone.from_config(
                    config["backbone_config"],
                    img_size=config["img_size"],
                    patch_size=patch_size,
                    in_chans=3,
                ),
                n_classes=N_CLASSES,
                head_dropout=0.3,
            ).to(device)

            # Load global backbone
            local_classifier.load_backbone_params(global_backbone)

            # Local optimizer (only backbone + head)
            local_optimizer = torch.optim.AdamW(
                local_classifier.parameters(),
                lr=config["lr"],
                weight_decay=0.01,
            )

            # Local training
            local_classifier.train()
            total_loss = 0
            n_batches = 0

            for epoch in range(config["local_epochs"]):
                for batch in loader:
                    images = batch[0].to(device)
                    labels = batch[1].to(device)

                    logits, features = local_classifier(images, return_features=True)

                    if prototype_bank is not None:
                        # Combined CE + contrastive loss
                        ce_loss = F.cross_entropy(logits, labels)
                        proto_out = prototype_bank(features, labels)
                        contrastive_loss = proto_out["contrastive_loss"]
                        loss = ce_loss + 0.5 * contrastive_loss
                    else:
                        loss = F.cross_entropy(logits, labels)

                    local_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_classifier.parameters(), 1.0)
                    local_optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            client_losses.append(avg_loss)
            client_params_list.append(local_classifier.get_backbone_params())
            client_sample_counts.append(len(loader.dataset))

            # Collect local prototype info
            if prototype_bank is not None:
                with torch.no_grad():
                    proto = prototype_bank.prototypes.data.clone()
                    client_prototypes.append(proto)

        # Server aggregation
        if aggregation == "fedavg":
            aggregated = fedavg_aggregate(client_params_list)
        elif aggregation == "volume_weighted":
            aggregated = volume_weighted_aggregate(
                client_params_list, client_sample_counts
            )
        elif aggregation == "prototype_aware" and prototype_bank is not None:
            aggregated = prototype_aware_aggregate(
                client_params_list,
                client_prototypes,
                prototype_bank.prototypes.data,
                alpha=0.5,
            )
        else:
            aggregated = fedavg_aggregate(client_params_list)

        # Update global model
        classifier.load_backbone_params(aggregated)

        # Update global prototypes
        if prototype_bank is not None and global_optimizer is not None:
            global_optimizer.step()

        # Evaluate
        classifier.eval()
        eval_results = evaluate_classifier(classifier, test_loader, device)

        round_time = time.time() - round_start

        history.append({
            "round": round_num + 1,
            "avg_client_loss": float(np.mean(client_losses)),
            "accuracy": eval_results["accuracy"],
            "macro_f1": eval_results["macro_f1"],
            "macro_precision": eval_results["macro_precision"],
            "macro_recall": eval_results["macro_recall"],
            "round_time": round_time,
        })

        # Print progress
        if (round_num + 1) % max(1, config["fl_rounds"] // 10) == 0 or round_num == 0:
            print(
                f"    Round {round_num+1:3d}/{config['fl_rounds']}  "
                f"Loss: {np.mean(client_losses):.4f}  "
                f"Acc: {eval_results['accuracy']:.4f}  "
                f"F1: {eval_results['macro_f1']:.4f}  "
                f"({round_time:.1f}s)"
            )

    # Final evaluation
    classifier.eval()
    final_results = evaluate_classifier(classifier, test_loader, device)

    return {
        "aggregation": aggregation,
        "history": history,
        "final_results": final_results,
        "params_info": params_info,
    }


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Medical-FL Framework")
    parser.add_argument(
        "--mode", type=str, default="quick",
        choices=list(CONFIGS.keys()),
        help="Experiment mode",
    )
    args = parser.parse_args()

    config = CONFIGS[args.mode]

    print("=" * 60)
    print("  Medical-FL: Federated Learning for Medical Imaging")
    print("=" * 60)
    print(f"  Mode: {args.mode} — {config['description']}")
    print(f"  Clients: {config['n_clients']}")
    print(f"  FL Rounds: {config['fl_rounds']}")
    print(f"  MAE Pretraining: {config['use_mae']} ({config['mae_epochs']} epochs)")
    print(f"  Prototype Contrastive: {config['use_prototype']}")
    print(f"  Aggregation: {config['aggregation']}")
    print(f"  Noise Ratio: {config['noise_ratio']}")
    print(f"  Non-IID Alpha: {config['noniid_alpha']}")

    start_time = time.time()

    if args.mode == "compare":
        # Compare all aggregation strategies
        results = {}

        # Phase 1: MAE pretraining (shared)
        pretrained = None
        if config["use_mae"] and config["mae_epochs"] > 0:
            pretrain_result = run_mae_pretraining(config)
            pretrained = pretrain_result["global_encoder"]

        # Phase 2: Compare aggregations
        for agg in ["fedavg", "volume_weighted", "prototype_aware"]:
            result = run_federated_training(config, pretrained, aggregation=agg)
            results[agg] = result

        # Print comparison
        print("\n" + "=" * 60)
        print("  Aggregation Strategy Comparison")
        print("=" * 60)
        print(f"  {'Strategy':20s} {'Accuracy':>8s} {'Macro F1':>8s} {'Best F1':>8s}")
        print("  " + "-" * 50)
        for agg, res in results.items():
            best_f1 = max(h["macro_f1"] for h in res["history"])
            print(
                f"  {agg:20s} "
                f"{res['final_results']['accuracy']:8.4f} "
                f"{res['final_results']['macro_f1']:8.4f} "
                f"{best_f1:8.4f}"
            )

        result = {"comparison": {k: {
            "final": v["final_results"],
            "best_f1": max(h["macro_f1"] for h in v["history"]),
            "history": v["history"],
        } for k, v in results.items()}}

    else:
        # Single experiment
        pretrained = None
        if config["use_mae"] and config["mae_epochs"] > 0:
            pretrain_result = run_mae_pretraining(config)
            pretrained = pretrain_result["global_encoder"]

        result = run_federated_training(
            config,
            pretrained_encoder=pretrained,
            aggregation=config["aggregation"] if config["aggregation"] != "all" else "fedavg",
        )

        # Print final results
        print("\n" + "=" * 60)
        print("  Final Results")
        print("=" * 60)
        print(format_results(result["final_results"], PLANE_NAMES))

    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.1f}s")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    saveable = {}
    for k, v in result.items():
        try:
            json.dumps({k: convert(v)})
            saveable[k] = convert(v)
        except (TypeError, ValueError):
            saveable[k] = str(v)

    result_path = results_dir / f"{args.mode}_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(saveable, f, indent=2, default=convert)
    print(f"  Results saved to: {result_path}")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
