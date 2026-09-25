"""Dataset loading, transforms, and batching."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from tqdm import tqdm


class CarDetectionDataset(Dataset):
    def __init__(self, yaml_path, split, img_size=None, transforms=None):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
        base_path = os.path.join(yaml_dir, cfg.get("path", ".") or ".")

        img_subdir = cfg[split]
        self.img_dir = os.path.normpath(os.path.join(base_path, img_subdir))
        self.label_dir = self.img_dir.replace("images", "labels")

        names = cfg["names"]
        if isinstance(names, dict):
            self.class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
        else:
            self.class_names = list(names)
        self.num_classes = int(cfg.get("nc", len(self.class_names)))
        self.img_size = img_size
        self.transforms = transforms

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        self.img_files = sorted(
            f for f in os.listdir(self.img_dir) if f.lower().endswith(valid_ext)
        )

    def __len__(self):
        return len(self.img_files)

    def _label_path_for(self, img_filename):
        stem = os.path.splitext(img_filename)[0]
        return os.path.join(self.label_dir, stem + ".txt")

    def __getitem__(self, idx):
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        boxes, labels = [], []
        label_path = self._label_path_for(img_filename)
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls_id_str, xc, yc, w, h = line.split()
                    cls_id = int(cls_id_str) + 1
                    xc, yc, w, h = map(float, (xc, yc, w, h))
                    x1 = (xc - w / 2) * orig_w
                    y1 = (yc - h / 2) * orig_h
                    x2 = (xc + w / 2) * orig_w
                    y2 = (yc + h / 2) * orig_h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)

        if self.img_size is not None:
            image = image.resize((self.img_size, self.img_size))
            sx, sy = self.img_size / orig_w, self.img_size / orig_h
            boxes = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes]

        image_tensor = TF.to_tensor(image)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        if self.transforms:
            image_tensor, target = self.transforms(image_tensor, target)
        return image_tensor, target


class CachedDataset(Dataset):
    """Cache decoded images in memory; apply transforms on each access."""

    def __init__(self, base_dataset, transforms=None):
        self.samples = []
        for i in tqdm(range(len(base_dataset)), desc="caching"):
            image, target = base_dataset[i]
            self.samples.append(((image * 255).to(torch.uint8), target))
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, target = self.samples[idx]
        image = image.float() / 255
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def get_train_transforms():
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=180),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.RandomPhotometricDistort(p=0.3),
        v2.SanitizeBoundingBoxes(),
    ])


def apply_transform(transform, image_tensor, target):
    img_h, img_w = image_tensor.shape[-2], image_tensor.shape[-1]
    boxes = tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=(img_h, img_w))
    image_tv = tv_tensors.Image(image_tensor)
    sample = {"boxes": boxes, "labels": target["labels"]}
    image_out, sample_out = transform(image_tv, sample)
    new_target = dict(target)
    new_target["boxes"] = torch.as_tensor(sample_out["boxes"], dtype=torch.float32).reshape(-1, 4)
    new_target["labels"] = torch.as_tensor(sample_out["labels"], dtype=torch.int64)
    return torch.as_tensor(image_out, dtype=torch.float32), new_target


class TrainAugment:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        return apply_transform(self.transform, image, target)
