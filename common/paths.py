"""Path resolution for local and Google Drive / Colab layouts."""
from __future__ import annotations

import os
from pathlib import Path

_DRIVE_CANDIDATES = (
    "/content/drive/MyDrive/vehicle detection",
    "/content/drive/MyDrive/vehicle-detection",
)


def project_root() -> Path:
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    for candidate in _DRIVE_CANDIDATES:
        p = Path(candidate)
        if p.is_dir():
            return p.resolve()

    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return project_root() / "data" / "dataset"


def dataset_yaml() -> Path:
    yaml_path = data_dir() / "dataset.yaml"
    if yaml_path.is_file():
        return yaml_path
    alt = data_dir() / "config.yaml"
    if alt.is_file():
        return alt
    return yaml_path


def weights_dir(model_name: str) -> Path:
    d = project_root() / model_name / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d
