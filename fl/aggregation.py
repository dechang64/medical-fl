"""Aggregation strategies: FedAvg, FedProx, FedBN, SCAFFOLD."""
import torch.nn as nn
from typing import Dict, List, Optional

class Aggregator:
    def aggregate(self, updates, weights=None): raise NotImplementedError

class FedAvg(Aggregator):
    def aggregate(self, updates, weights=None):
        if weights is None: weights = [1.0/len(updates)] * len(updates)
        total = sum(weights); weights = [w/total for w in weights]
        return {k: sum(w*u[k] for w, u in zip(weights, updates)) for k in updates[0]}

class FedProx(Aggregator):
    def aggregate(self, updates, weights=None): return FedAvg().aggregate(updates, weights)

class FedBN(Aggregator):
    def aggregate(self, updates, weights=None):
        if weights is None: weights = [1.0/len(updates)] * len(updates)
        total = sum(weights); weights = [w/total for w in weights]
        return {k: updates[0][k] if ("bn" in k.lower() or "norm" in k.lower()) else sum(w*u[k] for w, u in zip(weights, updates)) for k in updates[0]}

class ScaffoldAggregator(Aggregator):
    def aggregate(self, updates, weights=None): return FedAvg().aggregate(updates, weights)

def build_aggregator(strategy):
    return {"fedavg": FedAvg, "fedprox": FedProx, "fedbn": FedBN, "scaffold": ScaffoldAggregator}[strategy]()
