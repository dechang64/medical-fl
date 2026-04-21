"""Medical datasets: real images + synthetic generator + Non-IID partitioning."""
import os, random, numpy as np
from pathlib import Path
from typing import Optional, List, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T

class MedicalImageDataset(Dataset):
    """Directory-based: data_root/class_0/img001.png ..."""
    def __init__(self, data_root, transform=None, max_per_class=None):
        self.transform = transform; self.samples = []
        root = Path(data_root)
        self.class_names = sorted([d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}
        for cn in self.class_names:
            imgs = sorted((root / cn).glob("*"))
            if max_per_class: imgs = imgs[:max_per_class]
            for p in imgs:
                self.samples.append((str(p), self.class_to_idx[cn]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return {"image": img, "label": label}

class SyntheticMedicalDataset(Dataset):
    """Synthetic medical images with class-specific geometric patterns."""
    CLASS_NAMES = ["brain", "heart", "abdomen", "bpd", "femur", "face"]
    COLORS = [[0.15,0.25,0.55],[0.55,0.15,0.20],[0.20,0.50,0.25],[0.50,0.40,0.15],[0.35,0.20,0.50],[0.55,0.35,0.20]]

    def __init__(self, num_samples=600, num_classes=6, img_size=64, noise_ratio=0.0, seed=42, normalize=False):
        self.num_samples = num_samples; self.num_classes = num_classes
        self.img_size = img_size; self.normalize = normalize
        self.rng = np.random.RandomState(seed)
        self.images = []; self.labels = []; self.noisy_labels = []; self.is_noisy = []
        for i in range(num_samples):
            true_label = i % num_classes
            self.images.append(self._generate(true_label))
            self.labels.append(true_label)
            if noise_ratio > 0 and random.Random(seed + i).random() < noise_ratio:
                nl = random.Random(seed + i).choice([l for l in range(num_classes) if l != true_label])
                self.noisy_labels.append(nl); self.is_noisy.append(True)
            else:
                self.noisy_labels.append(true_label); self.is_noisy.append(False)

    def _generate(self, cls):
        s = self.img_size; rng = self.rng
        img = np.zeros((s, s, 3), dtype=np.float32)
        img[:] = np.array(self.COLORS[cls % len(self.COLORS)]) + rng.randn(s, s, 3) * 0.03
        y, x = np.ogrid[:s, :s]
        if cls == 0:  # concentric circles
            for r in range(10, s // 2, 8):
                cx, cy = s//2 + rng.randint(-5,5), s//2 + rng.randint(-5,5)
                img[((x-cx)**2+(y-cy)**2) < r**2] += 0.08
        elif cls == 1:  # cross
            w = rng.randint(s//6, s//4)
            img[s//2-w:s//2+w, s//4:3*s//4] += 0.12
            img[s//4:3*s//4, s//2-w:s//2+w] += 0.12
        elif cls == 2:  # ellipse
            cx, cy = s//2+rng.randint(-10,10), s//2+rng.randint(-10,10)
            rx, ry = rng.randint(s//4,s//3), rng.randint(s//5,s//4)
            img[((x-cx)/rx)**2+((y-cy)/ry)**2 < 1] += 0.10
        elif cls == 3:  # parallel lines
            off = rng.randint(s//6, s//4)
            for dy in [-off, off]:
                yp = s//2+dy
                img[max(0,yp-2):min(s,yp+2), s//6:5*s//6] += 0.15
        elif cls == 4:  # diagonal
            for t in np.linspace(0, 1, s):
                px, py = int(s*0.2+t*s*0.6), int(s*0.3+t*s*0.4)
                if 0<=px<s and 0<=py<s:
                    img[max(0,py-2):min(s,py+2), max(0,px-2):min(s,px+2)] += 0.15
        elif cls == 5:  # triangle
            sz = rng.randint(s//4, s//3)
            for i in range(3):
                a = i*2*np.pi/3 - np.pi/2
                ex, ey = int(s//2+sz*np.cos(a)), int(s//2+sz*np.sin(a))
                steps = max(abs(ex-s//2), abs(ey-s//2), 1)
                for t in np.linspace(0, 1, steps):
                    px, py = int(s//2+t*(ex-s//2)), int(s//2+t*(ey-s//2))
                    if 0<=px<s and 0<=py<s:
                        img[max(0,py-1):min(s,py+1), max(0,px-1):min(s,px+1)] += 0.12
        img += rng.randn(s, s, 3) * 0.04
        img *= rng.uniform(0.85, 1.15); img += rng.uniform(-0.05, 0.05)
        img = np.clip(img, 0, 1).transpose(2, 0, 1)
        t = torch.from_numpy(img)
        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            t = (t - mean) / std
        return t

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx], "noisy_label": self.noisy_labels[idx], "is_noisy": self.is_noisy[idx]}

def partition_non_iid(dataset, num_clients, alpha=0.5, seed=42):
    rng = np.random.RandomState(seed)
    targets = np.array([dataset[i]["label"] if isinstance(dataset[i], dict) else dataset[i][1] for i in range(len(dataset))])
    nc = targets.max() + 1
    client_indices = [[] for _ in range(num_clients)]
    for c in range(nc):
        idx = rng.permutation(np.where(targets == c)[0])
        props = rng.dirichlet(np.repeat(alpha, num_clients))
        props = (props * len(idx)).astype(int); start = 0
        for cid in range(num_clients):
            end = start + props[cid]
            client_indices[cid].extend(idx[start:end].tolist()); start = end
    return [Subset(dataset, idx) for idx in client_indices]

def create_federated_dataloaders(dataset, num_clients, batch_size=32, alpha=0.5, seed=42):
    subsets = partition_non_iid(dataset, num_clients, alpha, seed)
    return [DataLoader(s, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True) for s in subsets]
