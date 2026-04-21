"""Training logger with console + JSON."""
import os, json
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, name="medical_fl", log_dir="runs"):
        self.name = name; self.log_dir = Path(log_dir) / name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {}; self.entries = []
    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        pfx = {"INFO":"[INFO]","SUCCESS":"[OK]","WARNING":"[WARN]","ERROR":"[ERR]"}.get(level,"[LOG]")
        print(f"{pfx} {msg}")
    def log_metrics(self, metrics, step, prefix=""):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                fk = f"{prefix}/{k}" if prefix else k
                self.history.setdefault(fk, []).append({"step": step, "value": v})
        self.entries.append({"step": step, "timestamp": datetime.now().isoformat(), **metrics})
    def log_round(self, r, m):
        self.log(f"Round {r:3d} | Loss: {m.get('loss',0):.4f} | Acc: {m.get('accuracy',0):.4f} | Clients: {m.get('num_clients',0)}")
        self.log_metrics(m, step=r, prefix="fl")
    def save_history(self):
        with open(self.log_dir / "metrics.json", "w") as f: json.dump(self.history, f, indent=2)
        with open(self.log_dir / "log.json", "w") as f: json.dump(self.entries, f, indent=2)
    def close(self): self.save_history()
