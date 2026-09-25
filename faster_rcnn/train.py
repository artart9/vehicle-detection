"""Train Faster R-CNN; promote demo.pt on best val mAP."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.checkpoint import maybe_promote
from common.dataset import (
    CachedDataset,
    CarDetectionDataset,
    TrainAugment,
    collate_fn,
    get_train_transforms,
)
from common.metrics import detection_report, evaluate, faster_rcnn_complexity, format_map
from common.paths import dataset_yaml, weights_dir
from common.runtime import add_runtime_arg, loader_settings
from faster_rcnn.model import build_faster_rcnn, patch_focal_loss

IMG_SIZE = 640
NUM_EPOCHS = 50
LR = 1e-4


def grad_group(name: str):
    if "custom" in name or ".fpn." in name:
        return "fpn"
    if any(k in name for k in (".stem.", ".layer1.", ".layer2.", ".layer3.", ".layer4.")):
        return "backbone"
    if "rpn" in name:
        return "rpn"
    if "roi_heads" in name or "box_head" in name or "box_predictor" in name:
        return "box_head"
    return None


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN")
    add_runtime_arg(parser)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--max-batches", type=int, default=0, help="Cap train batches/epoch (0=all)")
    args = parser.parse_args()
    cfg = loader_settings("faster_rcnn", args.runtime)
    epochs = args.epochs
    img_size = args.img_size

    patch_focal_loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | runtime: {cfg.name} | batch={cfg.batch_size} workers={cfg.num_workers}")

    yaml_path = dataset_yaml()
    if not yaml_path.is_file():
        raise SystemExit(f"Dataset yaml missing: {yaml_path}")

    train_tf = get_train_transforms()
    train_ds = CarDetectionDataset(yaml_path, "train", img_size=img_size)
    val_ds = CarDetectionDataset(yaml_path, "val", img_size=img_size)
    train_cached = CachedDataset(train_ds, transforms=TrainAugment(train_tf))
    train_loader = DataLoader(
        train_cached, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=device.type == "cuda",
    )
    print(f"classes: {train_ds.class_names} | train={len(train_ds)} val={len(val_ds)}")

    model = build_faster_rcnn(num_classes=train_ds.num_classes).to(device)
    faster_rcnn_complexity(model, img_size=img_size, device=device.type)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(len(train_loader), 1)
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    loss_keys = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    grad_groups = ["backbone", "fpn", "rpn", "box_head"]
    wdir = weights_dir("faster_rcnn")

    for epoch in range(1, epochs + 1):
        model.train()
        running, parts = 0.0, {k: 0.0 for k in loss_keys}
        grad_sq = {g: 0.0 for g in grad_groups}
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)

            batch_sq = {g: 0.0 for g in grad_groups}
            for name, param in model.named_parameters():
                g = grad_group(name)
                if param.grad is None or g is None:
                    continue
                batch_sq[g] += param.grad.float().norm() ** 2
            if torch.isfinite(torch.as_tensor(sum(batch_sq.values()))):
                for g in grad_groups:
                    grad_sq[g] += batch_sq[g]

            scaler.step(optimizer)
            scaler.update()
            if epoch == 1:
                warmup.step()

            running += losses.item()
            for k in loss_keys:
                if k in loss_dict:
                    parts[k] += loss_dict[k].item()
            n_batches += 1
            if args.max_batches and n_batches >= args.max_batches:
                break

        avg_loss = running / max(n_batches, 1)
        val = evaluate(model, val_loader, device)
        val_map = float(val["map"].item())
        map_50 = float(val["map_50"].item()) if "map_50" in val else None

        total_sq = float(sum(grad_sq.values())) or 1.0
        print(f"Epoch {epoch}/{epochs} | train_loss: {avg_loss:.4f} | {format_map(val)}")
        print("   " + " | ".join(f"{k}: {parts[k]/n_batches:.4f}" for k in loss_keys if parts[k]))
        print(
            "   grad share: "
            + " | ".join(f"{g}: {100 * grad_sq[g] / total_sq:.1f}%" for g in grad_groups)
        )
        det = detection_report(model, val_loader, device)
        maybe_promote(
            wdir, model, val_map, epoch,
            extras={
                "map_50": map_50,
                "train_loss": avg_loss,
                "precision": det["precision"],
                "recall": det["recall"],
                "f1": det["f1"],
            },
        )


if __name__ == "__main__":
    main()
