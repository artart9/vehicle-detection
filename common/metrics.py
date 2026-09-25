"""Evaluation and complexity reporting."""
from __future__ import annotations

import copy
import math

import torch
from torch.utils.flop_counter import FlopCounterMode
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def evaluate(model, data_loader, device):
    was_training = model.training
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(preds_cpu, targets_cpu)
    result = metric.compute()
    if was_training:
        model.train()
    return result


def format_map(result) -> str:
    parts = []
    for k in ("map", "map_50", "map_75"):
        if k in result:
            v = result[k]
            parts.append(f"{k}: {v.item() if hasattr(v, 'item') else float(v):.4f}")
    return " | ".join(parts)


def _get_score_thresh(model):
    if hasattr(model, "score_thresh"):
        return model.score_thresh, "yolo"
    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "score_thresh"):
        return model.roi_heads.score_thresh, "frcnn"
    return None, None


def _set_score_thresh(model, value, kind):
    if kind == "yolo":
        model.score_thresh = value
    elif kind == "frcnn":
        model.roi_heads.score_thresh = value


@torch.no_grad()
def detection_report(model, data_loader, device, score_thresh=0.25, iou_thresh=0.5, n_bins=10):
    """Precision, recall, F1, count error, and score calibration (ECE)."""
    from torchvision.ops import box_iou

    was_training = model.training
    model.eval()
    old_thresh, kind = _get_score_thresh(model)
    if kind:
        _set_score_thresh(model, 0.001, kind)

    tp_total = fp_total = fn_total = n_img = 0
    tp_ious, count_errors, all_scores, all_tp = [], [], [], []

    for images, targets in data_loader:
        preds = model([img.to(device) for img in images])
        for p, t in zip(preds, targets):
            gt = t["boxes"].to(device)
            order = p["scores"].argsort(descending=True)
            boxes, scores = p["boxes"][order], p["scores"][order]
            tp = torch.zeros(len(boxes), dtype=torch.bool, device=device)
            match_iou = torch.zeros(len(boxes), device=device)
            if len(gt) and len(boxes):
                ious = box_iou(boxes, gt)
                taken = torch.zeros(len(gt), dtype=torch.bool, device=device)
                for i in range(len(boxes)):
                    iou, j = ious[i].masked_fill(taken, 0).max(0)
                    if iou >= iou_thresh:
                        tp[i], taken[j], match_iou[i] = True, True, iou

            keep = scores >= score_thresh
            n_tp, n_pred = int(tp[keep].sum()), int(keep.sum())
            tp_total += n_tp
            fp_total += n_pred - n_tp
            fn_total += len(gt) - n_tp
            tp_ious.append(match_iou[keep & tp])
            count_errors.append(abs(n_pred - len(gt)))
            all_scores.append(scores)
            all_tp.append(tp)
            n_img += 1

    if kind and old_thresh is not None:
        _set_score_thresh(model, old_thresh, kind)
    if was_training:
        model.train()

    n_img = max(n_img, 1)
    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    tp_ious = torch.cat(tp_ious) if tp_ious else torch.zeros(0)
    mean_iou = tp_ious.mean().item() if len(tp_ious) else 0.0

    scores = torch.cat(all_scores) if all_scores else torch.zeros(0, device=device)
    tp = torch.cat(all_tp).float() if all_tp else torch.zeros(0, device=device)
    edges = torch.linspace(0, 1, n_bins + 1, device=device if scores.numel() else "cpu")
    ece = 0.0
    print("--- calibration (score bin -> actual hit rate) ---")
    if scores.numel():
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = (scores >= lo) & (scores < hi)
            if int(b.sum()) < 5:
                continue
            ece += b.float().mean().item() * abs(scores[b].mean().item() - tp[b].mean().item())
            print(
                f"   {lo:.1f}-{hi:.1f}: mean score {scores[b].mean():.2f} | "
                f"hit rate {tp[b].mean():.2f} | n={int(b.sum())}"
            )
    else:
        print("   (no predictions)")

    print(f"--- at score >= {score_thresh}, IoU >= {iou_thresh} ---")
    print(f"   precision {precision:.3f} | recall {recall:.3f} | F1 {f1:.3f}")
    print(f"   per image: {fp_total / n_img:.2f} false positives | {fn_total / n_img:.2f} missed cars")
    print(f"   mean IoU of true positives: {mean_iou:.3f}")
    print(f"   count error (MAE): {sum(count_errors) / n_img:.2f} cars/image")
    print(f"   ECE: {ece:.3f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "ece": ece,
    }


def count_params(module):
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def gflops(fn, *args):
    with FlopCounterMode(display=False) as fc:
        fn(*args)
    return fc.get_total_flops() / 1e9


def _print_complexity_rows(rows, total_p, total_f, img_size):
    print(f"--- complexity @ {img_size}x{img_size}, batch 1 ---")
    for name, p, f in rows:
        pct_p = 100 * p / total_p if total_p else 0
        pct_f = 100 * f / total_f if total_f else 0
        print(f"{name:12s} {p/1e6:6.2f}M params ({pct_p:4.1f}%) | {f:6.2f} GFLOPs ({pct_f:4.1f}%)")
    print(f"{'total':12s} {total_p/1e6:6.2f}M params          | {total_f:6.2f} GFLOPs")
    print(f"weights on disk: fp32 {total_p*4/1e6:.1f} MB | fp16 {total_p*2/1e6:.1f} MB | int8 {total_p/1e6:.1f} MB")


@torch.no_grad()
def yolo_complexity(model, img_size=640, device="cuda"):
    m = copy.deepcopy(model).eval().to(device)
    x = torch.randn(1, 3, img_size, img_size, device=device)
    xn = (x - m.mean) / m.std
    feats = m.backbone(xn)
    neck_feats = m.neck(feats)
    blocks = [
        ("backbone", m.backbone, lambda: m.backbone(xn)),
        ("neck", m.neck, lambda: m.neck(feats)),
        ("head", m.head, lambda: m.head(neck_feats)),
    ]
    rows, total_p, total_f = [], 0, 0.0
    for name, module, run in blocks:
        p, f = count_params(module), gflops(run)
        rows.append((name, p, f))
        total_p += p
        total_f += f
    _print_complexity_rows(rows, total_p, total_f, img_size)
    print("feature maps:")
    for f, s in zip(feats, m.strides):
        print(f"   P{int(math.log2(s))} (stride {s}): {tuple(f.shape)}")
    print(f"candidate cells before NMS: {sum(f.shape[-2] * f.shape[-1] for f in feats)}")
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        m(x)
        torch.cuda.synchronize()
        print(f"peak activation memory (fp32, batch 1): {(torch.cuda.max_memory_allocated() - base)/1e6:.1f} MB")
    del m
    return {"params_M": total_p / 1e6, "gflops": total_f}


@torch.no_grad()
def faster_rcnn_complexity(model, img_size=640, device="cuda"):
    m = copy.deepcopy(model).eval().to(device)
    x = torch.randn(1, 3, img_size, img_size, device=device)
    feats = m.backbone(x)

    def run_backbone():
        return m.backbone(x)

    rows = [
        ("backbone", count_params(m.backbone), gflops(run_backbone)),
        ("rpn", count_params(m.rpn), 0.0),
        ("roi_heads", count_params(m.roi_heads), 0.0),
    ]

    def full_forward():
        return m([x[0]])

    total_forward = gflops(full_forward)
    backbone_f = rows[0][2]
    remainder = max(total_forward - backbone_f, 0.0)
    rpn_p, roi_p = rows[1][1], rows[2][1]
    split = rpn_p + roi_p or 1
    rows[1] = ("rpn", rpn_p, remainder * rpn_p / split)
    rows[2] = ("roi_heads", roi_p, remainder * roi_p / split)

    total_p = sum(r[1] for r in rows)
    total_f = sum(r[2] for r in rows)
    _print_complexity_rows(rows, total_p, total_f, img_size)
    print("feature maps:")
    if isinstance(feats, dict):
        for k, v in feats.items():
            print(f"   {k}: {tuple(v.shape)}")
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        m([x[0]])
        torch.cuda.synchronize()
        print(f"peak activation memory (fp32, batch 1): {(torch.cuda.max_memory_allocated() - base)/1e6:.1f} MB")
    del m
    return {"params_M": total_p / 1e6, "gflops": total_f}
