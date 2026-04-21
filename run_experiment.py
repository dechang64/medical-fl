#!/usr/bin/env python3
"""
Full Experiment Runner.

Runs the complete pipeline:
1. MAE self-supervised pretraining
2. Federated fine-tuning with prototype-aware aggregation
3. Evaluation

Compares different configurations:
- With/without MAE pretraining
- FedAvg vs FedProx vs Prototype-Aware aggregation
- Different noise ratios
- Different Non-IID levels

Usage:
    python run_experiment.py --config configs/default.yaml
    python run_experiment.py --quick  # Quick test with reduced settings
"""

import argparse
import yaml
import json
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Full Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (fewer epochs/rounds)")
    parser.add_argument("--skip_pretrain", action="store_true", help="Skip MAE pretraining")
    parser.add_argument("--skip_federated", action="store_true", help="Skip federated training")
    parser.add_argument("--backbone", type=str, default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    return parser.parse_args()


def load_config(args):
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def run_command(cmd, description):
    """Run a command and log output."""
    print(f"\n{'='*60}")
    print(f"[EXPERIMENT] {description}")
    print(f"[EXPERIMENT] Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[ERROR] {description} failed with code {result.returncode}")
        return False

    print(f"\n[EXPERIMENT] {description} completed in {elapsed:.1f}s")
    return True


def main():
    args = parse_args()
    config = load_config(args)

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    results = {}
    python = sys.executable

    # Quick mode overrides
    quick_args = []
    if args.quick:
        quick_args = [
            "--epochs", "10",
            "--rounds", "5",
            "--num_samples", "200",
            "--num_clients", "3",
        ]

    # ─── Experiment 1: MAE Pretraining ───
    if not args.skip_pretrain:
        pretrain_path = str(output_dir / "mae_encoder.pth")
        cmd = [
            python, "train_pretrain.py",
            "--config", args.config,
            "--backbone", args.backbone,
            "--save_path", pretrain_path,
            "--log_dir", str(output_dir / "logs" / "mae"),
        ] + quick_args

        success = run_command(cmd, "MAE Self-Supervised Pretraining")
        results["pretrain"] = {"success": success, "path": pretrain_path}
    else:
        pretrain_path = None
        results["pretrain"] = {"success": False, "path": None}

    # ─── Experiment 2: Federated Training (multiple configs) ───
    if not args.skip_federated:
        experiment_configs = [
            {
                "name": "fedavg_no_pretrain",
                "aggregation": "fedavg",
                "pretrained": None,
                "extra": ["--no_prototype"],
            },
            {
                "name": "fedavg_pretrained",
                "aggregation": "fedavg",
                "pretrained": pretrain_path,
                "extra": ["--no_prototype"],
            },
            {
                "name": "prototype_aware_no_pretrain",
                "aggregation": "prototype_aware",
                "pretrained": None,
                "extra": [],
            },
            {
                "name": "prototype_aware_pretrained",
                "aggregation": "prototype_aware",
                "pretrained": pretrain_path,
                "extra": [],
            },
        ]

        for exp in experiment_configs:
            save_path = str(output_dir / f"model_{exp['name']}.pth")
            cmd = [
                python, "train_federated.py",
                "--config", args.config,
                "--backbone", args.backbone,
                "--aggregation", exp["aggregation"],
                "--save_path", save_path,
                "--log_dir", str(output_dir / "logs" / exp["name"]),
            ] + exp["extra"] + quick_args

            if exp["pretrained"]:
                cmd.extend(["--pretrained", exp["pretrained"]])

            success = run_command(cmd, f"Federated: {exp['name']}")
            results[exp["name"]] = {"success": success, "path": save_path}

            # Evaluate
            if success:
                eval_cmd = [
                    python, "evaluate.py",
                    "--checkpoint", save_path,
                    "--config", args.config,
                    "--num_samples", "500",
                ]
                run_command(eval_cmd, f"Evaluation: {exp['name']}")

    # ─── Summary ───
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"\nResults:")
    for name, result in results.items():
        status = "✓" if result["success"] else "✗"
        print(f"  {status} {name}: {result.get('path', 'N/A')}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
