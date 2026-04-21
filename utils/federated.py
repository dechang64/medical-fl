"""
Federated Learning utilities for medical imaging.

Includes:
1. FedAvg aggregation with task-aware weighting
2. Prototype contrastive aggregation
3. Communication cost estimation
4. Evaluation metrics for medical imaging
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
#  FedAvg Aggregation
# ═══════════════════════════════════════════════════════════════

def fedavg_aggregate(
    client_params: List[Dict[str, torch.Tensor]],
    client_weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Federated Averaging aggregation.

    Args:
        client_params: List of parameter dicts from each client
        client_weights: Optional weights (e.g., by data size)

    Returns:
        Aggregated parameter dict
    """
    if client_weights is None:
        client_weights = [1.0 / len(client_params)] * len(client_params)
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]

    aggregated = {}
    for key in client_params[0].keys():
        aggregated[key] = sum(
            w * params[key] for w, params in zip(client_weights, client_params)
        )

    return aggregated


def volume_weighted_aggregate(
    client_params: List[Dict[str, torch.Tensor]],
    client_sample_counts: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Volume-weighted aggregation (UltraFedFM style).

    Clients with more data get higher weight.

    Args:
        client_params: List of parameter dicts
        client_sample_counts: Number of samples per client

    Returns:
        Aggregated parameter dict
    """
    total_samples = sum(client_sample_counts)
    weights = [n / total_samples for n in client_sample_counts]
    return fedavg_aggregate(client_params, weights)


def prototype_aware_aggregate(
    client_params: List[Dict[str, torch.Tensor]],
    client_prototypes: List[torch.Tensor],
    global_prototype: torch.Tensor,
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Prototype-aware aggregation.

    Clients whose local prototypes are closer to the global prototype
    get higher aggregation weight. This down-weights clients with
    drifted or noisy data.

    Args:
        client_params: List of parameter dicts
        client_prototypes: [n_clients, embed_dim] local prototypes
        global_prototype: [embed_dim] global prototype
        alpha: Blending factor (0 = pure FedAvg, 1 = pure prototype-weighted)

    Returns:
        Aggregated parameter dict
    """
    # Compute prototype distances
    distances = []
    for proto in client_prototypes:
        dist = F.cosine_similarity(
            proto.unsqueeze(0), global_prototype.unsqueeze(0)
        ).item()
        distances.append(dist)

    # Convert distances to weights (higher similarity = higher weight)
    distances = np.array(distances)
    weights = np.exp(distances * 5)  # Temperature scaling
    weights = weights / weights.sum()

    # Blend with uniform weights
    uniform = np.ones(len(weights)) / len(weights)
    blended = alpha * weights + (1 - alpha) * uniform
    blended = blended / blended.sum()

    return fedavg_aggregate(client_params, blended.tolist())


# ═══════════════════════════════════════════════════════════════
#  Parameter Utilities
# ═══════════════════════════════════════════════════════════════

def get_model_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get model parameters as a dict."""
    return {k: v.clone() for k, v in model.state_dict().items()}


def set_model_params(model: nn.Module, params: Dict[str, torch.Tensor]):
    """Load parameters into model."""
    model.load_state_dict(params)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def communication_cost(params: Dict[str, torch.Tensor]) -> dict:
    """
    Estimate communication cost.

    Returns:
        dict with bytes, KB, MB
    """
    total_bytes = sum(v.numel() * 4 for v in params.values())  # float32
    return {
        "bytes": total_bytes,
        "KB": total_bytes / 1024,
        "MB": total_bytes / (1024 ** 2),
    }


# ═══════════════════════════════════════════════════════════════
#  Evaluation Metrics
# ═══════════════════════════════════════════════════════════════

def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Evaluate a classifier on a dataloader.

    Returns:
        dict with accuracy, precision, recall, F1 (macro), per-class metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()

    # Per-class metrics
    n_classes = len(np.unique(all_labels))
    per_class = {}

    for c in range(n_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        per_class[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((all_labels == c).sum()),
        }

    # Macro averages
    macro_precision = np.mean([per_class[c]["precision"] for c in per_class])
    macro_recall = np.mean([per_class[c]["recall"] for c in per_class])
    macro_f1 = np.mean([per_class[c]["f1"] for c in per_class])

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "n_samples": len(all_labels),
    }


def format_results(results: dict, class_names: list = None) -> str:
    """Format evaluation results as a readable string."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(results["per_class"]))]

    lines = []
    lines.append(f"  Accuracy:       {results['accuracy']:.4f}")
    lines.append(f"  Macro Precision: {results['macro_precision']:.4f}")
    lines.append(f"  Macro Recall:    {results['macro_recall']:.4f}")
    lines.append(f"  Macro F1:        {results['macro_f1']:.4f}")
    lines.append(f"  Samples:         {results['n_samples']}")
    lines.append("")
    lines.append(f"  {'Class':25s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'Support':>7s}")
    lines.append("  " + "-" * 55)

    for i, name in enumerate(class_names):
        pc = results["per_class"][i]
        lines.append(
            f"  {name:25s} {pc['precision']:6.4f} {pc['recall']:6.4f} "
            f"{pc['f1']:6.4f} {pc['support']:7d}"
        )

    return "\n".join(lines)
