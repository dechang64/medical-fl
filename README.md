# Medical-FL

Privacy-preserving federated learning framework for medical imaging.

## Quick Start

```bash
pip install -r requirements.txt

# Demo (~60s on CPU, no GPU needed)
python scripts/demo.py

# Federated training
python scripts/train_federated.py --num_clients 5 --rounds 20

# MAE self-supervised pretraining
python scripts/train_pretrain.py --epochs 50

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/model.pth
```

## Architecture

```
medical-fl/
├── models/          # ViT, MAE, Prototype Bank, Task Heads, UNet
├── data/            # Dataset loading, synthetic data, transforms
├── fl/              # FL Server, Client, Aggregation strategies
├── utils/           # Metrics (Dice/IoU/AUC/F1), Logger
├── configs/         # YAML configuration
└── scripts/         # Training & evaluation scripts
```

## Key Features

| Feature | Description |
|---------|-------------|
| ViT Backbone | Tiny (2.9M) / Small (21.7M) / Base (85.8M) |
| MAE Pretraining | Federated self-supervised learning, no labels needed |
| Prototype Contrastive | Robust to noisy labels, handles Non-IID data |
| Multi-Task Heads | Classification, Segmentation, Detection |
| Flexible Aggregation | FedAvg, FedProx, FedBN, SCAFFOLD, Prototype-Aware |
| Synthetic Data | Built-in generator for testing without real data |

## Demo Results (CPU, synthetic data)

```
3 clients x 15 rounds x 3 local epochs | noise=10% | Non-IID alpha=0.5
R14/15 | Acc: 1.0000 | ##############################
```

## References

- ViT: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
- MAE: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- Prototype FL: Moccia et al., "Contrastive prototype federated learning against noisy labels", Int J CARS, 2025
