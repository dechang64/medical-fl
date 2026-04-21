"""FL client: local training, prototype updates, model upload."""
import copy, time
from typing import Dict, Optional
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from models.prototype import PrototypeBank

@dataclass
class ClientConfig:
    client_id: str; local_epochs: int = 3; batch_size: int = 32
    learning_rate: float = 1e-3; weight_decay: float = 5e-5
    fedprox_mu: float = 0.01; gradient_clip: float = 1.0; device: str = "cpu"

class FLClient:
    def __init__(self, model, config, train_loader, val_loader=None, prototype_bank=None, criterion=None):
        self.model = model.to(config.device); self.config = config
        self.train_loader = train_loader; self.val_loader = val_loader
        self.prototype_bank = prototype_bank
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.global_params = {k: v.clone() for k, v in self.model.state_dict().items()}

    def receive_global_model(self, state_dict):
        self.model.load_state_dict(state_dict, strict=False)
        self.global_params = {k: v.clone() for k, v in state_dict.items()}

    def train_round(self):
        self.model.train()
        total_loss = 0.0; total_correct = 0; total_samples = 0
        start = time.time()
        for _ in range(self.config.local_epochs):
            for batch in self.train_loader:
                imgs, labels = batch["image"], batch["label"]
                imgs, labels = imgs.to(self.config.device), labels.to(self.config.device)
                self.optimizer.zero_grad()
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
                # FedProx regularization
                if self.config.fedprox_mu > 0:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if name in self.global_params:
                            prox_loss += ((param - self.global_params[name].to(self.config.device)) ** 2).sum()
                    loss += self.config.fedprox_mu / 2 * prox_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += imgs.size(0)
        # Update prototypes
        if self.prototype_bank:
            self.prototype_bank.eval()
            with torch.no_grad():
                for batch in self.train_loader:
                    feats = self.model[0].forward_features(batch["image"].to(self.config.device))
                    self.prototype_bank(feats, batch["label"].to(self.config.device))
        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        from fl.server import ClientUpdate
        return ClientUpdate(client_id=self.config.client_id, round_num=0, num_samples=total_samples,
            local_loss=avg_loss, local_accuracy=avg_acc,
            params={k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            prototype_state={k: v.cpu().clone() for k, v in self.prototype_bank.state_dict().items()} if self.prototype_bank else None,
            training_time=time.time() - start)

class MAEClient:
    def __init__(self, mae, config, train_loader):
        self.mae = mae.to(config.device); self.config = config; self.train_loader = train_loader
        self.optimizer = torch.optim.AdamW(self.mae.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    def receive_global_params(self, params): self.mae.load_state_dict(params, strict=False)
    def train_round(self):
        self.mae.train(); total_loss = 0.0; total = 0; start = time.time()
        for _ in range(self.config.local_epochs):
            for batch in self.train_loader:
                imgs = batch["image"].to(self.config.device)
                loss, _, _ = self.mae(imgs)
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mae.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                total_loss += loss.item() * imgs.size(0); total += imgs.size(0)
        from fl.server import ClientUpdate
        return ClientUpdate(client_id=self.config.client_id, round_num=0, num_samples=total,
            local_loss=total_loss/max(total,1), local_accuracy=0.0,
            params={k: v.cpu().clone() for k, v in self.mae.state_dict().items() if k.startswith("encoder.")},
            training_time=time.time()-start)
