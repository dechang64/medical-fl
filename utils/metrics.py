"""Metrics: accuracy, F1, AUC, Dice, IoU, confusion matrix."""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from collections import defaultdict

class MetricsTracker:
    def __init__(self, num_classes, task="classification"):
        self.num_classes = num_classes; self.task = task; self.reset()
    def reset(self):
        self.all_preds = []; self.all_targets = []; self.all_probs = []
        self.total_loss = 0.0; self.total_samples = 0
    def update(self, predictions, targets, loss=None, probabilities=None):
        if predictions.dim() > 1:
            preds = predictions.argmax(1); probs = F.softmax(predictions, dim=1)
        else:
            preds = predictions; probs = probabilities
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_targets.extend(targets.cpu().numpy().tolist())
        if probs is not None: self.all_probs.extend(probs.detach().cpu().numpy().tolist())
        if loss is not None: self.total_loss += loss * targets.size(0)
        self.total_samples += targets.size(0)
    def compute(self):
        preds = np.array(self.all_preds); targets = np.array(self.all_targets)
        accuracy = (preds == targets).mean()
        metrics = {"accuracy": accuracy, "loss": self.total_loss / max(self.total_samples, 1), "num_samples": self.total_samples}
        # Per-class metrics
        for c in range(self.num_classes):
            tp = ((preds == c) & (targets == c)).sum()
            fp = ((preds == c) & (targets != c)).sum()
            fn = ((preds != c) & (targets == c)).sum()
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            metrics[f"class_{c}_precision"] = prec; metrics[f"class_{c}_recall"] = rec; metrics[f"class_{c}_f1"] = f1
        # Macro averages
        metrics["macro_precision"] = np.mean([metrics[f"class_{c}_precision"] for c in range(self.num_classes)])
        metrics["macro_recall"] = np.mean([metrics[f"class_{c}_recall"] for c in range(self.num_classes)])
        metrics["macro_f1"] = np.mean([metrics[f"class_{c}_f1"] for c in range(self.num_classes)])
        # AUC
        if self.all_probs and len(self.all_probs) == len(targets):
            try:
                from sklearn.metrics import roc_auc_score
                metrics["auc"] = roc_auc_score(targets, np.array(self.all_probs), multi_class="ovr", average="macro")
            except Exception: metrics["auc"] = 0.0
        return metrics

def compute_dice_score(pred, target, smooth=1e-6):
    p, t = pred.view(-1), target.view(-1)
    return ((p * t).sum() * 2 / (p.sum() + t.sum() + smooth)).item()

def compute_iou(pred, target, smooth=1e-6):
    p, t = pred.view(-1), target.view(-1)
    return ((p * t).sum() + smooth / (p.sum() + t.sum() - (p * t).sum() + smooth)).item()

def compute_confusion_matrix(preds, targets, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(preds, targets): cm[t][p] += 1
    return cm

def format_metrics(metrics, prefix=""):
    return "\n".join(f"{prefix}{k}: {v:.4f}" for k, v in sorted(metrics.items()) if isinstance(v, float) and not k.startswith("class_"))
