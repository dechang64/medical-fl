#!/usr/bin/env python3
"""Federated training with configurable parameters."""
import sys, os, argparse, time, yaml
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
from utils.logger import Logger

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backbone", default=None, choices=["tiny","small","base"])
    p.add_argument("--num_clients", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--local_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--aggregation", default=None, choices=["fedavg","fedprox","prototype_aware"])
    p.add_argument("--noise", type=float, default=None)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--save_path", default="checkpoints/model.pth")
    p.add_argument("--log_dir", default="runs/federated")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f: config = yaml.safe_load(f)
    backbone = args.backbone or config.get("model",{}).get("backbone","tiny")
    img_size = args.img_size or config.get("model",{}).get("img_size",64)
    nc = config.get("data",{}).get("num_classes",6)
    n_clients = args.num_clients or config.get("fl",{}).get("num_clients",5)
    nr = args.rounds or config.get("fl",{}).get("rounds",50)
    le = args.local_epochs or config.get("fl",{}).get("local_epochs",3)
    lr = args.lr or config.get("fl",{}).get("learning_rate",1e-3)
    bs = args.batch_size or config.get("fl",{}).get("batch_size",32)
    agg = args.aggregation or config.get("fl",{}).get("aggregation","prototype_aware")
    noise = args.noise if args.noise is not None else config.get("data",{}).get("noise_ratio",0.2)
    samples = config.get("data",{}).get("num_samples_per_client",200)
    alpha = config.get("data",{}).get("non_iid_alpha",0.5)

    logger = Logger(name="federated", log_dir=args.log_dir)
    logger.log(f"Config: backbone={backbone}, clients={n_clients}, rounds={nr}, lr={lr}, agg={agg}")

    train_ds = SyntheticMedicalDataset(num_samples=samples*n_clients, num_classes=nc, img_size=img_size, noise_ratio=noise, seed=42, normalize=True)
    val_ds = SyntheticMedicalDataset(num_samples=200, num_classes=nc, img_size=img_size, noise_ratio=0.0, seed=99, normalize=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    client_loaders = create_federated_dataloaders(train_ds, n_clients, batch_size=bs, alpha=alpha)

    bb = build_vit(backbone, img_size=img_size)
    model = nn.Sequential(bb, build_head("classification", in_dim=bb.out_dim, num_classes=nc))
    server = FLServer(model=model, prototype_bank=PrototypeBank(nc, bb.out_dim), aggregation=agg)

    clients = []
    for i in range(n_clients):
        b2 = build_vit(backbone, img_size=img_size)
        clients.append(FLClient(
            model=nn.Sequential(b2, build_head("classification", in_dim=b2.out_dim, num_classes=nc)),
            config=ClientConfig(client_id=f"c{i}", local_epochs=le, batch_size=bs, learning_rate=lr),
            train_loader=client_loaders[i], prototype_bank=PrototypeBank(nc, b2.out_dim)))

    def eval_fn(m):
        m.eval(); t = MetricsTracker(nc)
        with torch.no_grad():
            for b in val_loader: t.update(m(b["image"]).argmax(1), b["label"])
        r = t.compute(); return r["loss"], r["accuracy"]

    logger.log(f"Starting {nr} rounds of federated training...")
    for r in range(1, nr + 1):
        for c in clients:
            c.receive_global_model(server.model.state_dict())
            c.prototype_bank.load_state_dict(server.prototype_bank.state_dict())
        updates = [c.train_round() for c in clients]
        result = server.aggregate(updates)
        loss, acc = server.evaluate_global_model(eval_fn)
        logger.log_round(r, {"loss": loss, "accuracy": acc, "num_clients": result.num_participants})
        if r % 10 == 0:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({"model_state_dict": server.model.state_dict(), "round": r, "accuracy": acc}, args.save_path)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({"model_state_dict": server.model.state_dict(), "round": nr, "accuracy": acc}, args.save_path)
    logger.log(f"Done! Final acc: {acc:.4f}, saved to {args.save_path}", level="SUCCESS")
    logger.close()

if __name__ == "__main__":
    main()
