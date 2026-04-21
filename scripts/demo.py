#!/usr/bin/env python3
"""Quick demo: synthetic data + 3 clients + 10 rounds. ~90s on CPU."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from models.vit import build_vit
from models.heads import build_head
from models.prototype import PrototypeBank
from data.dataset import SyntheticMedicalDataset, create_federated_dataloaders
from fl.server import FLServer
from fl.client import FLClient, ClientConfig
from utils.metrics import MetricsTracker

def main():
    nc, bs, le, nr, n_clients, img_size = 6, 64, 2, 10, 3, 64

    print("=" * 60)
    print("  Medical-FL Demo")
    print("  Federated Learning for Medical Imaging")
    print("=" * 60)

    # Data
    print("\n[1/4] Generating synthetic data...")
    train_ds = SyntheticMedicalDataset(num_samples=200*n_clients, num_classes=nc, img_size=img_size, noise_ratio=0.1, seed=42, normalize=True)
    val_ds = SyntheticMedicalDataset(num_samples=200, num_classes=nc, img_size=img_size, noise_ratio=0.0, seed=99, normalize=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    client_loaders = create_federated_dataloaders(train_ds, n_clients, batch_size=bs, alpha=0.5)
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    print(f"  {n_clients} clients (Non-IID, alpha=0.5)")

    # Model
    print("\n[2/4] Building model...")
    bb = build_vit("tiny", img_size=img_size)
    model = nn.Sequential(bb, build_head("classification", in_dim=bb.out_dim, num_classes=nc))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ViT-Tiny + ClassificationHead: {n_params:,} params")

    # Clients
    print("\n[3/4] Creating clients...")
    clients = []
    for i in range(n_clients):
        b2 = build_vit("tiny", img_size=img_size)
        clients.append(FLClient(
            model=nn.Sequential(b2, build_head("classification", in_dim=b2.out_dim, num_classes=nc)),
            config=ClientConfig(client_id=f"client_{i}", local_epochs=le, batch_size=bs, learning_rate=3e-3),
            train_loader=client_loaders[i],
            prototype_bank=PrototypeBank(nc, b2.out_dim),
        ))

    # Train
    server = FLServer(model=model, prototype_bank=PrototypeBank(nc, bb.out_dim), aggregation="prototype_aware")
    print(f"\n[4/4] Training ({nr} rounds, {le} local epochs each)...")
    print("-" * 60)
    start = time.time()
    history = []

    def eval_fn(m):
        m.eval(); t = MetricsTracker(nc)
        with torch.no_grad():
            for b in val_loader:
                t.update(m(b["image"]).argmax(1), b["label"])
        r = t.compute(); return r["loss"], r["accuracy"]

    for r in range(1, nr + 1):
        for c in clients:
            c.receive_global_model(server.model.state_dict())
            c.prototype_bank.load_state_dict(server.prototype_bank.state_dict())
        updates = [c.train_round() for c in clients]
        result = server.aggregate(updates)
        loss, acc = server.evaluate_global_model(eval_fn)
        history.append(acc)
        bar = "#" * int(acc * 30)
        print(f"  R{r:2d}/{nr} | Loss: {result.global_loss:.4f} | Acc: {acc:.4f} | {bar} | {time.time()-start:.1f}s")

    print("-" * 60)
    best = max(history)
    print(f"\n  Final accuracy: {history[-1]*100:.1f}%")
    print(f"  Best accuracy:  {best*100:.1f}% (Round {history.index(best)+1})")
    print(f"  Total time:     {time.time()-start:.1f}s")
    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
