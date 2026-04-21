"""Prototype bank for contrastive federated learning."""
import torch, torch.nn as nn, torch.nn.functional as F, math
from typing import Optional, Dict

class PrototypeBank(nn.Module):
    def __init__(self, n_classes, embed_dim, temperature=0.1, momentum=0.9, confidence_threshold=0.7):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.02)
        self.register_buffer("class_counts", torch.zeros(n_classes))
        self.temperature = temperature; self.momentum = momentum
        self.confidence_threshold = confidence_threshold
        self.n_classes = n_classes; self.embed_dim = embed_dim
    def forward(self, features, labels, update=True):
        protos = F.normalize(self.prototypes, dim=1)
        feats = F.normalize(features, dim=1)
        logits = feats @ protos.t() / self.temperature
        loss = F.cross_entropy(logits, labels)
        if update: self._update_prototypes(feats, labels)
        return loss, logits
    def _update_prototypes(self, features, labels):
        with torch.no_grad():
            for c in range(self.n_classes):
                mask = labels == c
                if mask.sum() > 0:
                    new_proto = features[mask].mean(dim=0)
                    new_proto = F.normalize(new_proto, dim=0)
                    self.prototypes.data[c] = self.momentum * self.prototypes.data[c] + (1 - self.momentum) * new_proto
                    self.class_counts[c] += mask.sum().item()
    def get_contrastive_loss(self, features, labels):
        protos = F.normalize(self.prototypes, dim=1)
        feats = F.normalize(features, dim=1)
        pos = protos[labels]
        sim = feats @ protos.t() / self.temperature
        pos_sim = (feats * pos).sum(dim=1, keepdim=True) / self.temperature
        loss = -pos_sim + torch.logsumexp(sim, dim=1)
        return loss.mean()
    def filter_noisy(self, features, labels):
        with torch.no_grad():
            protos = F.normalize(self.prototypes, dim=1)
            feats = F.normalize(features, dim=1)
            sim = feats @ protos.t()
            max_sim, pred = sim.max(dim=1)
            clean_mask = (pred == labels) & (max_sim > self.confidence_threshold)
        return clean_mask

class PrototypeAwareAggregator:
    @staticmethod
    def compute_client_weights(client_prototypes, global_prototypes, temperature=0.1):
        sims = {}
        for cid, state in client_prototypes.items():
            p = F.normalize(state["prototypes"], dim=1)
            g = F.normalize(global_prototypes, dim=1)
            sims[cid] = torch.mm(p, g.t()).trace().item() / p.shape[0]
        mx = max(sims.values())
        exps = {k: math.exp((v - mx) / temperature) for k, v in sims.items()}
        total = sum(exps.values())
        return {k: v / total for k, v in exps.items()}
