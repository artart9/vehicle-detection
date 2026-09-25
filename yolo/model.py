"""Custom YOLO detector (ResNet backbone, FPN neck, decoupled head)."""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import (
    FrozenBatchNorm2d,
    batched_nms,
    clip_boxes_to_image,
    generalized_box_iou_loss,
    sigmoid_focal_loss,
)


class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, d=1, g=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, k, s, padding=d * (k - 1) // 2, dilation=d, groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.act = {"silu": nn.SiLU(inplace=True), "relu": nn.ReLU(inplace=True), None: nn.Identity()}[act]

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.bn, nn.Identity):
            return
        c, bn = self.conv, self.bn
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        fused = nn.Conv2d(
            c.in_channels, c.out_channels, c.kernel_size, c.stride,
            c.padding, c.dilation, c.groups, bias=True,
        ).to(c.weight.device, c.weight.dtype)
        fused.weight.copy_(c.weight * scale.view(-1, 1, 1, 1))
        fused.bias.copy_(bn.bias - bn.running_mean * scale)
        self.conv, self.bn = fused, nn.Identity()


class SPPF(nn.Module):
    def __init__(self, c_in, c_out, k=5, act="silu"):
        super().__init__()
        c_h = c_in // 2
        self.cv1 = ConvBNAct(c_in, c_h, 1, act=act)
        self.cv2 = ConvBNAct(c_h * 4, c_out, 1, act=act)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class YOLOResNetBackbone(nn.Module):
    def __init__(
        self,
        backbone_name="resnet34",
        cutoff_layer="layer3",
        trainable_pretrained_layers=1,
        extra_conv_layers=None,
        use_sppf=True,
    ):
        super().__init__()
        resnet = getattr(torchvision.models, backbone_name)(
            weights="DEFAULT", norm_layer=FrozenBatchNorm2d
        )
        layer_order = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
        if cutoff_layer not in ("layer3", "layer4"):
            raise ValueError("cutoff_layer must be 'layer3' or 'layer4'")
        kept = layer_order[: layer_order.index(cutoff_layer) + 1]
        self.pretrained_stem = nn.Sequential(
            OrderedDict((n, getattr(resnet, n)) for n in kept)
        )
        children = list(self.pretrained_stem.children())
        for child in children[: len(children) - trainable_pretrained_layers]:
            for p in child.parameters():
                p.requires_grad = False

        c5 = resnet.fc.in_features
        stage_channels = {"layer1": c5 // 8, "layer2": c5 // 4, "layer3": c5 // 2, "layer4": c5}
        self.tap_names = [n for n in ("layer2", "layer3", "layer4") if n in kept]
        level_channels = [stage_channels[n] for n in self.tap_names]

        if extra_conv_layers is None:
            extra_conv_layers = (
                [{"out_channels": 512, "kernel_size": 3, "stride": 2, "pool": None}]
                if cutoff_layer == "layer3"
                else [{"out_channels": 512, "kernel_size": 3, "stride": 1, "pool": None}]
            )

        blocks, in_c, total_stride = [], stage_channels[cutoff_layer], 1
        for spec in extra_conv_layers:
            stride = spec.get("stride", 1)
            blocks.append(
                ConvBNAct(in_c, spec["out_channels"], spec.get("kernel_size", 3),
                          s=stride, d=spec.get("dilation", 1), act="relu")
            )
            total_stride *= stride
            if spec.get("pool") == "max":
                blocks.append(nn.MaxPool2d(2, 2))
                total_stride *= 2
            elif spec.get("pool") == "avg":
                blocks.append(nn.AvgPool2d(2, 2))
                total_stride *= 2
            in_c = spec["out_channels"]

        self.custom_stack = nn.Sequential(*blocks)
        self.custom_creates_level = total_stride == 2
        if self.custom_creates_level:
            level_channels.append(in_c)
        elif total_stride == 1:
            level_channels[-1] = in_c
        else:
            raise ValueError(f"custom stack total stride must be 1 or 2, got {total_stride}")
        if len(level_channels) != 3:
            raise ValueError("need exactly 3 levels (P3/P4/P5)")

        self.sppf = SPPF(level_channels[-1], level_channels[-1], act="relu") if use_sppf else nn.Identity()
        self.out_channels = level_channels
        self.strides = [8, 16, 32]

    def forward(self, x):
        feats = []
        for name, layer in self.pretrained_stem.named_children():
            x = layer(x)
            if name in self.tap_names:
                feats.append(x)
        x = self.sppf(self.custom_stack(x))
        if self.custom_creates_level:
            feats.append(x)
        else:
            feats[-1] = x
        return feats


class YOLONeck(nn.Module):
    def __init__(self, in_channels, out_channels=128, act="silu"):
        super().__init__()
        self.lateral = nn.ModuleList(ConvBNAct(c, out_channels, 1, act=act) for c in in_channels)
        self.smooth = nn.ModuleList(ConvBNAct(out_channels, out_channels, 3, act=act) for _ in in_channels)
        self.out_channels = [out_channels] * len(in_channels)

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        return [s(x) for s, x in zip(self.smooth, lat)]


class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden=128, act="silu", prior_prob=0.01):
        super().__init__()

        def branch(c, out):
            return nn.Sequential(
                ConvBNAct(c, hidden, 3, act=act),
                ConvBNAct(hidden, hidden, 3, act=act),
                nn.Conv2d(hidden, out, 1),
            )

        self.cls_branch = nn.ModuleList(branch(c, num_classes) for c in in_channels)
        self.box_branch = nn.ModuleList(branch(c, 4) for c in in_channels)
        bias = -math.log((1 - prior_prob) / prior_prob)
        for b in self.cls_branch:
            nn.init.constant_(b[-1].bias, bias)

    def forward(self, feats):
        return [(c(f), b(f)) for c, b, f in zip(self.cls_branch, self.box_branch, feats)]


class YOLODetector(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_kwargs=None,
        neck_channels=128,
        head_hidden=128,
        size_ranges=((0, 64), (64, 128), (128, float("inf"))),
        center_radius=1.5,
        box_weight=2.0,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        pre_nms_topk=1000,
    ):
        super().__init__()
        self.backbone = YOLOResNetBackbone(**(backbone_kwargs or {}))
        self.neck = YOLONeck(self.backbone.out_channels, neck_channels)
        self.head = YOLOHead(self.neck.out_channels, num_classes, head_hidden)
        self.num_classes = num_classes
        self.strides = self.backbone.strides
        self.size_ranges = size_ranges
        self.center_radius = center_radius
        self.box_weight = box_weight
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.pre_nms_topk = pre_nms_topk
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images, targets=None):
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        img_hw = images.shape[-2:]
        outs = self.head(self.neck(self.backbone((images - self.mean) / self.std)))
        cls_logits, reg, points, strides, lo, hi = self._flatten(outs)
        if self.training:
            return self._loss(cls_logits, reg, points, strides, lo, hi, targets)
        return self._postprocess(cls_logits, reg, points, strides, img_hw)

    def _flatten(self, outs):
        cls_all, reg_all, pts_all, str_all, lo_all, hi_all = [], [], [], [], [], []
        for (cls, reg), s, (lo, hi) in zip(outs, self.strides, self.size_ranges):
            b, c, h, w = cls.shape
            cls_all.append(cls.permute(0, 2, 3, 1).reshape(b, h * w, c))
            reg_all.append(reg.permute(0, 2, 3, 1).reshape(b, h * w, 4))
            ys, xs = torch.meshgrid(
                torch.arange(h, device=cls.device),
                torch.arange(w, device=cls.device),
                indexing="ij",
            )
            pts_all.append((torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5) * s)
            str_all.append(torch.full((h * w,), float(s), device=cls.device))
            lo_all.append(torch.full((h * w,), float(lo), device=cls.device))
            hi_all.append(torch.full((h * w,), float(hi), device=cls.device))
        return (
            torch.cat(cls_all, 1).float(),
            torch.cat(reg_all, 1).float(),
            torch.cat(pts_all),
            torch.cat(str_all),
            torch.cat(lo_all),
            torch.cat(hi_all),
        )

    @staticmethod
    def _decode(points, strides, reg):
        d = F.softplus(reg) * strides[:, None]
        return torch.cat([points - d[..., :2], points + d[..., 2:]], dim=-1)

    def _assign(self, points, strides, lo, hi, gt):
        px, py = points[:, 0:1], points[:, 1:2]
        x1, y1, x2, y2 = gt.unbind(1)
        inside = torch.stack([px - x1, py - y1, x2 - px, y2 - py], -1).min(-1).values > 0
        r = self.center_radius * strides[:, None]
        near = ((px - (x1 + x2) / 2).abs() < r) & ((py - (y1 + y2) / 2).abs() < r)
        size = torch.maximum(x2 - x1, y2 - y1)
        in_range = (size >= lo[:, None]) & (size < hi[:, None])
        candidate = inside & near & in_range
        area = ((x2 - x1) * (y2 - y1)).expand_as(candidate)
        area = torch.where(candidate, area, torch.full_like(area, float("inf")))
        min_area, matched = area.min(dim=1)
        return matched, torch.isfinite(min_area)

    def _loss(self, cls_logits, reg, points, strides, lo, hi, targets):
        pred_boxes = self._decode(points, strides, reg)
        cls_targets = torch.zeros_like(cls_logits)
        loss_box = cls_logits.sum() * 0
        num_pos = 0
        for i, t in enumerate(targets):
            gt, labels = t["boxes"], t["labels"] - 1
            if len(gt) == 0:
                continue
            matched, pos = self._assign(points, strides, lo, hi, gt)
            pos_idx = pos.nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            cls_targets[i, pos_idx, labels[matched[pos_idx]]] = 1.0
            loss_box = loss_box + generalized_box_iou_loss(
                pred_boxes[i, pos_idx], gt[matched[pos_idx]], reduction="sum"
            )
            num_pos += len(pos_idx)
        num_pos = max(num_pos, 1)
        loss_cls = sigmoid_focal_loss(cls_logits, cls_targets, reduction="sum") / num_pos
        return {"loss_cls": loss_cls, "loss_box": self.box_weight * loss_box / num_pos}

    def _postprocess(self, cls_logits, reg, points, strides, img_hw):
        boxes_all = self._decode(points, strides, reg)
        results = []
        for boxes, logits in zip(boxes_all, cls_logits):
            scores, labels = logits.sigmoid().max(dim=1)
            keep = scores > self.score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if len(scores) > self.pre_nms_topk:
                top = scores.topk(self.pre_nms_topk).indices
                boxes, scores, labels = boxes[top], scores[top], labels[top]
            boxes = clip_boxes_to_image(boxes, img_hw)
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)[: self.detections_per_img]
            results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep] + 1,
            })
        return results
