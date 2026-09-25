"""Best-checkpoint tracking for demo weights."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def _metrics_path(weights_dir: Path) -> Path:
    return weights_dir / "best_metrics.json"


def _demo_path(weights_dir: Path) -> Path:
    return weights_dir / "demo.pt"


def load_best(weights_dir: Path) -> dict:
    path = _metrics_path(weights_dir)
    if not path.is_file():
        return {"metric": "val_map", "value": float("-inf"), "epoch": None, "map_50": None, "train_loss": None}
    with open(path) as f:
        return json.load(f)


def maybe_promote(
    weights_dir: Path,
    model: torch.nn.Module,
    val_map: float,
    epoch: int,
    extras: dict | None = None,
) -> bool:
    """Write demo.pt and best_metrics.json when val_map improves."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best = load_best(weights_dir)
    prev = float(best.get("value", float("-inf")))
    if val_map != val_map:
        return False
    if val_map <= prev:
        return False

    payload = {
        "metric": "val_map",
        "value": float(val_map),
        "epoch": int(epoch),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extras:
        for k, v in extras.items():
            if hasattr(v, "item"):
                v = v.item()
            payload[k] = float(v) if isinstance(v, (int, float)) else v

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_map": float(val_map),
            "metrics": payload,
        },
        _demo_path(weights_dir),
    )
    with open(_metrics_path(weights_dir), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  new best val_map={val_map:.4f} -> {_demo_path(weights_dir)}")
    return True


def load_demo(weights_dir: Path, model: torch.nn.Module, map_location="cpu"):
    path = _demo_path(weights_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"No demo weights at {path}. Train the model first to create a checkpoint."
        )
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt
