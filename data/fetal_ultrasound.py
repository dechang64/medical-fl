"""
Data loading for medical imaging federated learning.

Supports:
1. Synthetic data generation for testing (no real data needed)
2. Fetal ultrasound standard plane classification (6 classes)
3. Non-IID data partitioning for FL simulation
4. Noisy label injection for robustness testing

Fetal ultrasound standard planes (ISUOG guidelines):
  0: Transverse cerebellum (brain)
  1: Four-chamber view (heart)
  2: Abdominal circumference
  3: BPD/OFD (biparietal diameter)
  4: Femur length
  5: Profile (face/nose)

Reference:
  Moccia et al., Int J CARS 2025 (12,400 images, 6 planes)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Dict, List, Optional
from PIL import Image


# ═══════════════════════════════════════════════════════════════
#  Fetal Ultrasound Standard Plane Names
# ═══════════════════════════════════════════════════════════════

PLANE_NAMES = [
    "Transverse Cerebellum",
    "Four-Chamber View",
    "Abdominal Circumference",
    "Biparietal Diameter",
    "Femur Length",
    "Facial Profile",
]

N_CLASSES = len(PLANE_NAMES)


# ═══════════════════════════════════════════════════════════════
#  Synthetic Data Generation
# ═══════════════════════════════════════════════════════════════

class SyntheticFetalUltrasound(Dataset):
    """
    Synthetic fetal ultrasound dataset for testing.

    Generates random grayscale images with class-specific patterns
    to simulate different ultrasound planes. Each class has a
    distinct visual pattern:
    - Different texture frequencies
    - Different brightness distributions
    - Different spatial structures

    This allows testing the full FL pipeline without real medical data.

    Args:
        n_samples: Number of samples to generate
        img_size: Image size (square)
        n_classes: Number of classes
        noise_ratio: Fraction of labels to corrupt (for noise robustness testing)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        n_samples: int = 1000,
        img_size: int = 224,
        n_classes: int = N_CLASSES,
        noise_ratio: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_classes = n_classes
        self.noise_ratio = noise_ratio

        rng = np.random.RandomState(seed)

        # Generate images and labels
        self.images = torch.zeros(n_samples, 3, img_size, img_size)
        self.labels = torch.zeros(n_samples, dtype=torch.long)

        for i in range(n_samples):
            label = rng.randint(0, n_classes)
            self.labels[i] = label
            self.images[i] = self._generate_image(label, rng)

        # Inject noisy labels
        if noise_ratio > 0:
            n_noisy = int(n_samples * noise_ratio)
            noisy_indices = rng.choice(n_samples, n_noisy, replace=False)
            for idx in noisy_indices:
                # Random wrong label
                wrong_label = rng.randint(0, n_classes)
                while wrong_label == self.labels[idx]:
                    wrong_label = rng.randint(0, n_classes)
                self.labels[idx] = wrong_label

    def _generate_image(self, label: int, rng: np.random.RandomState) -> torch.Tensor:
        """
        Generate a synthetic ultrasound-like image for a given class.

        Each class has a distinct pattern:
        - Class 0 (Brain): circular structure in center
        - Class 1 (Heart): four-chamber pattern
        - Class 2 (Abdomen): elliptical structure
        - Class 3 (BPD): bilateral symmetric structure
        - Class 4 (Femur): elongated horizontal structure
        - Class 5 (Profile): vertical asymmetric structure
        """
        img_size = self.img_size
        # Base: dark background with speckle noise (ultrasound-like)
        img = rng.randn(img_size, img_size) * 0.15 + 0.1

        # Add class-specific structure
        y, x = np.ogrid[:img_size, :img_size]
        cx, cy = img_size // 2, img_size // 2

        if label == 0:  # Brain: concentric circles
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            img += 0.5 * np.exp(-((r - 40) ** 2) / 200)
            img += 0.3 * np.exp(-((r - 20) ** 2) / 100)
            img += 0.2 * np.exp(-(r ** 2) / 800)

        elif label == 1:  # Heart: four-chamber cross
            img += 0.5 * np.exp(-(((x - cx) ** 2) / 400 + ((y - cy) ** 2) / 800))
            img += 0.3 * np.exp(-(((x - cx) ** 2) / 800 + ((y - cy) ** 2) / 400))
            # Septum
            img += 0.2 * np.exp(-((x - cx) ** 2) / 50) * (np.abs(y - cy) < 30)
            img += 0.2 * np.exp(-((y - cy) ** 2) / 50) * (np.abs(x - cx) < 30)

        elif label == 2:  # Abdomen: ellipse
            r = np.sqrt(((x - cx) / 1.3) ** 2 + ((y - cy) / 0.8) ** 2)
            img += 0.5 * np.exp(-((r - 50) ** 2) / 300)
            img += 0.3 * np.exp(-(r ** 2) / 1500)

        elif label == 3:  # BPD: bilateral ovals
            for dx in [-30, 30]:
                r = np.sqrt((x - cx - dx) ** 2 + (y - cy) ** 2)
                img += 0.5 * np.exp(-((r - 25) ** 2) / 150)
                img += 0.3 * np.exp(-(r ** 2) / 500)

        elif label == 4:  # Femur: elongated
            img += 0.5 * np.exp(-(((x - cx) ** 2) / 3000 + ((y - cy) ** 2) / 200))
            img += 0.3 * np.exp(-(((x - cx) ** 2) / 1500 + ((y - cy) ** 2) / 100))

        elif label == 5:  # Profile: vertical structure
            img += 0.5 * np.exp(-(((x - cx + 10) ** 2) / 300 + ((y - cy) ** 2) / 2000))
            img += 0.3 * np.exp(-(((x - cx - 20) ** 2) / 200 + ((y - cy + 30) ** 2) / 500))

        # Add speckle noise (ultrasound characteristic)
        img += rng.randn(img_size, img_size) * 0.1

        # Clip and normalize
        img = np.clip(img, 0, 1)

        # Convert to 3-channel (grayscale -> RGB)
        img_rgb = np.stack([img, img, img], axis=0)

        return torch.from_numpy(img_rgb).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════
#  Non-IID Data Partitioning
# ═══════════════════════════════════════════════════════════════

def partition_noniid(
    dataset: Dataset,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[Subset]:
    """
    Partition dataset across clients with Dirichlet Non-IID distribution.

    Each client gets a different class distribution, controlled by alpha:
    - alpha -> 0: extreme Non-IID (each client has mostly 1-2 classes)
    - alpha -> inf: IID (uniform distribution)

    Args:
        dataset: Full dataset
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter
        seed: Random seed

    Returns:
        List of Subset, one per client
    """
    rng = np.random.RandomState(seed)
    n_samples = len(dataset)
    labels = np.array([dataset[i][1] for i in range(n_samples)])
    n_classes = len(np.unique(labels))

    # Dirichlet distribution for each class
    client_indices = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)

        # Split this class's samples across clients using Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions * len(class_indices)).astype(int)

        # Distribute remaining samples
        remainder = len(class_indices) - proportions.sum()
        for i in range(remainder):
            proportions[i % n_clients] += 1

        # Assign indices
        start = 0
        for i in range(n_clients):
            end = start + proportions[i]
            client_indices[i].extend(class_indices[start:end].tolist())
            start = end

    # Shuffle each client's indices
    for i in range(n_clients):
        rng.shuffle(client_indices[i])

    return [Subset(dataset, indices) for indices in client_indices]


def partition_quantity_skew(
    dataset: Dataset,
    n_clients: int,
    skew_factor: float = 0.5,
    seed: int = 42,
) -> List[Subset]:
    """
    Partition with quantity skew (different clients have different data sizes).

    Simulates real-world scenario where large hospitals have more data
    than small clinics.

    Args:
        dataset: Full dataset
        n_clients: Number of clients
        skew_factor: How skewed (0 = uniform, 1 = extreme)
        seed: Random seed

    Returns:
        List of Subset, one per client
    """
    rng = np.random.RandomState(seed)
    n_samples = len(dataset)

    # Generate sample counts using exponential distribution
    raw_counts = rng.exponential(1, n_clients)
    raw_counts = raw_counts / raw_counts.sum() * n_samples

    # Apply skew
    if skew_factor > 0:
        max_count = raw_counts.max()
        min_count = raw_counts.min()
        target_ratio = 1 - skew_factor * 0.8  # e.g., skew=0.5 -> ratio=0.6
        raw_counts = raw_counts ** (np.log(target_ratio) / np.log(min_count / max_count))
        raw_counts = raw_counts / raw_counts.sum() * n_samples

    counts = raw_counts.astype(int)
    counts[-1] = n_samples - counts[:-1].sum()  # ensure total matches

    # Assign indices
    all_indices = rng.permutation(n_samples)
    subsets = []
    start = 0
    for i in range(n_clients):
        end = start + counts[i]
        subsets.append(Subset(dataset, all_indices[start:end].tolist()))
        start = end

    return subsets


# ═══════════════════════════════════════════════════════════════
#  Federated Data Loading
# ═══════════════════════════════════════════════════════════════

def get_federated_dataloaders(
    n_clients: int = 5,
    samples_per_client: int = 200,
    img_size: int = 224,
    batch_size: int = 16,
    noise_ratio: float = 0.2,
    noniid_alpha: float = 0.5,
    quantity_skew: float = 0.3,
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Create federated data loaders for FL simulation.

    Args:
        n_clients: Number of simulated clients/hospitals
        samples_per_client: Average samples per client
        img_size: Image size
        batch_size: Batch size for training
        noise_ratio: Label noise ratio (simulates annotation disagreement)
        noniid_alpha: Dirichlet alpha for Non-IID partitioning
        quantity_skew: Data quantity skew between clients
        seed: Random seed

    Returns:
        client_loaders: List of DataLoaders, one per client
        test_loader: Test DataLoader (IID, no noise)
    """
    # Create full dataset
    total_samples = n_clients * samples_per_client
    full_dataset = SyntheticFetalUltrasound(
        n_samples=total_samples,
        img_size=img_size,
        noise_ratio=noise_ratio,
        seed=seed,
    )

    # Non-IID partition
    client_subsets = partition_noniid(
        full_dataset, n_clients, alpha=noniid_alpha, seed=seed
    )

    # Create data loaders
    client_loaders = [
        DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        for subset in client_subsets
    ]

    # Test set (IID, no noise)
    test_dataset = SyntheticFetalUltrasound(
        n_samples=200,
        img_size=img_size,
        noise_ratio=0.0,
        seed=seed + 100,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return client_loaders, test_loader


def print_data_stats(client_loaders: List[DataLoader], n_classes: int = N_CLASSES):
    """Print data distribution statistics across clients."""
    print("\n" + "=" * 60)
    print("Data Distribution Across Clients")
    print("=" * 60)

    for i, loader in enumerate(client_loaders):
        labels = []
        for batch in loader:
            if len(batch) == 3:
                labels.extend(batch[1].numpy().tolist())
            else:
                labels.extend(batch[1].numpy().tolist())

        counts = np.bincount(labels, minlength=n_classes)
        total = len(labels)
        print(f"\n  Client {i}: {total} samples")
        for c, count in enumerate(counts):
            bar = "#" * int(count / max(counts.max(), 1) * 20)
            print(f"    {PLANE_NAMES[c]:25s} {count:4d} ({count/total*100:5.1f}%) {bar}")

    print("\n" + "=" * 60)
