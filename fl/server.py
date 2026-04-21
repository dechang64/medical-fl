"""FL server: aggregation, prototype sync, round orchestration."""
import copy, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import torch, torch.nn as nn
from models.prototype import PrototypeBank, PrototypeAwareAggregator

@dataclass
class ClientUpdate:
    client_id: str; round_num: int; num_samples: int
    local_loss: float; local_accuracy: float
    params: Dict[str, torch.Tensor]
    prototype_state: Optional[Dict[str, torch.Tensor]] = None
    training_time: float = 0.0

@dataclass
class AggregationResult:
    round_num: int; num_participants: int; global_loss: float; global_accuracy: float
    aggregation_weights: Dict[str, float]; communication_cost_mb: float; training_time: float

@dataclass
class RoundHistory:
    rounds: List[AggregationResult] = field(default_factory=list)
    def add(self, r): self.rounds.append(r)
    def get_latest(self): return self.rounds[-1] if self.rounds else None
    def summary(self):
        if not self.rounds: return {}
        return {"total_rounds": len(self.rounds), "best_accuracy": max(r.global_accuracy for r in self.rounds),
                "final_accuracy": self.rounds[-1].global_accuracy, "total_comm_mb": sum(r.communication_cost_mb for r in self.rounds)}

class FLServer:
    def __init__(self, model, prototype_bank=None, aggregation="prototype_aware"):
        self.model = model; self.prototype_bank = prototype_bank
        self.aggregation = aggregation; self.current_round = 0
        self.history = RoundHistory(); self.registered_clients = []
        self.client_contributions = defaultdict(float)

    def register_client(self, client_id):
        if client_id not in self.registered_clients:
            self.registered_clients.append(client_id)

    def aggregate(self, updates: List[ClientUpdate]) -> AggregationResult:
        self.current_round += 1
        weights = self._compute_weights(updates)
        new_params = {}
        for key in updates[0].params:
            new_params[key] = sum(weights[u.client_id] * u.params[key] for u in updates)
        self.model.load_state_dict(new_params, strict=False)
        # Aggregate prototypes via FedAvg
        if self.prototype_bank and all(u.prototype_state for u in updates):
            for key in ["prototypes", "class_counts"]:
                if key in updates[0].prototype_state:
                    self.prototype_bank.state_dict()[key].data.copy_(
                        sum(weights[u.client_id] * u.prototype_state[key] for u in updates))
        avg_loss = sum(u.local_loss * weights[u.client_id] for u in updates)
        avg_acc = sum(u.local_accuracy * weights[u.client_id] for u in updates)
        comm = sum(sum(p.numel() * 4 for p in u.params.values()) / 1e6 for u in updates)
        result = AggregationResult(round_num=self.current_round, num_participants=len(updates),
            global_loss=avg_loss, global_accuracy=avg_acc, aggregation_weights=weights,
            communication_cost_mb=comm, training_time=sum(u.training_time for u in updates))
        self.history.add(result)
        for u in updates: self.client_contributions[u.client_id] += u.num_samples
        return result

    def _compute_weights(self, updates):
        if self.aggregation == "prototype_aware" and self.prototype_bank and all(u.prototype_state for u in updates):
            return PrototypeAwareAggregator.compute_client_weights(
                {u.client_id: u.prototype_state for u in updates}, self.prototype_bank.prototypes.data)
        return self._fedavg_weights(updates)

    def _fedavg_weights(self, updates):
        total = sum(u.num_samples for u in updates)
        return {u.client_id: u.num_samples / total for u in updates}

    def evaluate_global_model(self, eval_fn):
        self.model.eval(); return eval_fn(self.model)

    def get_summary(self):
        return {"current_round": self.current_round, "num_clients": len(self.registered_clients),
                "aggregation": self.aggregation, "history": self.history.summary()}
