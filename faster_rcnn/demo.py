"""Run Faster R-CNN inference with demo.pt."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.checkpoint import load_best, load_demo
from common.dataset import CarDetectionDataset
from common.paths import dataset_yaml, weights_dir
from faster_rcnn.model import build_faster_rcnn, patch_focal_loss

IMG_SIZE = 640
SCORE_THRESH = 0.35


def main():
    patch_focal_loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_path = dataset_yaml()
    if not yaml_path.is_file():
        raise SystemExit(f"Dataset yaml missing: {yaml_path}")

    val_ds = CarDetectionDataset(yaml_path, "val", img_size=IMG_SIZE)
    model = build_faster_rcnn(num_classes=val_ds.num_classes).to(device)
    wdir = weights_dir("faster_rcnn")
    ckpt = load_demo(wdir, model, map_location=device)
    best = load_best(wdir)
    print(f"loaded demo.pt (epoch={ckpt.get('epoch')}, val_map={best.get('value')})")

    model.eval()
    idx = random.randint(0, len(val_ds) - 1)
    image_tensor, target = val_ds[idx]
    with torch.no_grad():
        pred = model([image_tensor.to(device)])[0]

    keep = pred["scores"] > SCORE_THRESH
    pred_boxes = pred["boxes"][keep].cpu()
    pred_labels = pred["labels"][keep].cpu()
    pred_scores = pred["scores"][keep].cpu()
    pred_strs = [
        f"{val_ds.class_names[l - 1]} {s:.2f}"
        for l, s in zip(pred_labels.tolist(), pred_scores.tolist())
    ]
    gt_boxes = target["boxes"]
    gt_strs = [val_ds.class_names[l - 1] for l in target["labels"].tolist()]

    print(f"val image #{idx} — {val_ds.img_files[idx]}")
    print("GT:", list(zip(gt_boxes.tolist(), gt_strs)) or "(none)")
    print("Pred:", list(zip(pred_boxes.tolist(), pred_strs)) or "(none)")

    drawn = (image_tensor * 255).to(torch.uint8)
    if len(gt_boxes):
        drawn = draw_bounding_boxes(drawn, gt_boxes, labels=gt_strs, colors="blue", width=3)
    if len(pred_boxes):
        drawn = draw_bounding_boxes(drawn, pred_boxes, labels=pred_strs, colors="red", width=3)

    plt.figure(figsize=(8, 8))
    plt.imshow(drawn.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Blue=GT | Red=pred")
    plt.tight_layout()
    out = wdir / "demo_preview.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
