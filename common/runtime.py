"""DataLoader presets for local and Colab runs."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class LoaderSettings:
    batch_size: int
    num_workers: int
    name: str


_PRESETS = {
    "yolo": {
        "local": LoaderSettings(batch_size=8, num_workers=0, name="local"),
        "colab": LoaderSettings(batch_size=32, num_workers=2, name="colab"),
    },
    "faster_rcnn": {
        "local": LoaderSettings(batch_size=4, num_workers=0, name="local"),
        "colab": LoaderSettings(batch_size=12, num_workers=2, name="colab"),
    },
}


def is_colab() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        return True
    if os.path.exists("/content") and "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_runtime(choice: str = "auto") -> str:
    choice = (choice or "auto").lower()
    if choice == "auto":
        env = os.environ.get("RUNTIME", "").lower()
        if env in ("local", "colab"):
            return env
        return "colab" if is_colab() else "local"
    if choice not in ("local", "colab"):
        raise ValueError(f"runtime must be auto|local|colab, got {choice!r}")
    return choice


def loader_settings(model: str, runtime: str = "auto") -> LoaderSettings:
    rt = resolve_runtime(runtime)
    if model not in _PRESETS:
        raise KeyError(f"unknown model preset {model!r}")
    return _PRESETS[model][rt]


def add_runtime_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime",
        choices=("auto", "local", "colab"),
        default="auto",
        help="DataLoader batch size and num_workers preset",
    )
