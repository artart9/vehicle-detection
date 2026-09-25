"""Custom Faster R-CNN with ResNet multi-scale features and FPN."""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops import FeaturePyramidNetwork, FrozenBatchNorm2d, MultiScaleRoIAlign
import torchvision.models.detection.roi_heads as roi_heads_module


class CustomBackboneWithFPN(nn.Module):
    """ResNet stages, per-level conv refinement, then FPN (strides 8/16/32)."""

    def __init__(
        self,
        backbone_name="resnet50",
        trainable_pretrained_layers=1,
        fpn_out_channels=256,
        extra_channels=256,
    ):
        super().__init__()
        resnet = getattr(torchvision.models, backbone_name)(
            weights="DEFAULT", norm_layer=FrozenBatchNorm2d
        )
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        children = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        freeze_until = max(0, len(children) - trainable_pretrained_layers)
        for child in children[:freeze_until]:
            for p in child.parameters():
                p.requires_grad = False

        c5 = resnet.fc.in_features
        in_channels_list = [c5 // 8, c5 // 4, c5 // 2, c5]
        stage_channels = in_channels_list[1:]
        self.custom = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, extra_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(extra_channels),
                nn.ReLU(inplace=True),
            )
            for c in stage_channels
        ])
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[extra_channels] * 3,
            out_channels=fpn_out_channels,
        )
        self.out_channels = fpn_out_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        feats = OrderedDict()
        x = self.layer2(x)
        feats["0"] = self.custom[0](x)
        x = self.layer3(x)
        feats["1"] = self.custom[1](x)
        x = self.layer4(x)
        feats["2"] = self.custom[2](x)
        return self.fpn(feats)


def build_faster_rcnn(
    num_classes,
    backbone_name="resnet50",
    trainable_pretrained_layers=1,
    fpn_out_channels=256,
    rpn_conv_depth=1,
    box_head_hidden_dims=(256,),
    anchor_sizes=((32,), (50,), (70,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 3,
    roi_output_size=7,
    dropout=0.0,
):
    backbone = CustomBackboneWithFPN(
        backbone_name=backbone_name,
        trainable_pretrained_layers=trainable_pretrained_layers,
        fpn_out_channels=fpn_out_channels,
    )
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2"],
        output_size=roi_output_size,
        sampling_ratio=2,
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    rpn_head = RPNHead(backbone.out_channels, num_anchors, conv_depth=rpn_conv_depth)

    in_channels = backbone.out_channels * roi_output_size ** 2
    layers = [nn.Flatten(start_dim=1)]
    prev = in_channels
    for h in box_head_hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        prev = h
    box_head = nn.Sequential(*layers)
    box_head.out_channels = prev
    box_predictor = FastRCNNPredictor(prev, num_classes + 1)

    return FasterRCNN(
        backbone,
        num_classes=None,
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        box_roi_pool=roi_pooler,
        box_head=box_head,
        box_predictor=box_predictor,
    )


def focal_loss(class_logits, labels, gamma=2.0, alpha=0.25):
    ce = F.cross_entropy(class_logits, labels, reduction="none")
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()


def custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = focal_loss(class_logits, labels)

    pos = torch.where(labels > 0)[0]
    labels_pos = labels[pos]
    n = class_logits.shape[0]
    box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[pos, labels_pos],
        regression_targets[pos],
        beta=1 / 9,
        reduction="sum",
    ) / labels.numel()
    return classification_loss, box_loss


def patch_focal_loss():
    roi_heads_module.fastrcnn_loss = custom_fastrcnn_loss
