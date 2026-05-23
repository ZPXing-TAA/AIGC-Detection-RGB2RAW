from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
from torchvision import models


class Identity(nn.Module):
    def forward(self, x):
        return x


def _make_resnet50(pretrained: bool = True) -> nn.Module:
    try:
        return models.resnet50(pretrained=pretrained)
    except TypeError:
        weights = "IMAGENET1K_V1" if pretrained else None
        return models.resnet50(weights=weights)


def adapt_resnet_conv1_to_4ch(resnet: nn.Module) -> nn.Module:
    old_conv = resnet.conv1
    new_conv = nn.Conv2d(
        4,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        old_weight = old_conv.weight.data
        new_conv.weight.data[:, :3, :, :] = old_weight
        new_conv.weight.data[:, 3:4, :, :] = old_weight.mean(dim=1, keepdim=True)
    resnet.conv1 = new_conv
    return resnet


class ResNet50Classifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, pretrained: bool = True):
        super(ResNet50Classifier, self).__init__()
        self.in_channels = int(in_channels)
        self.backbone = _make_resnet50(pretrained=pretrained)
        if self.in_channels == 4:
            self.backbone = adapt_resnet_conv1_to_4ch(self.backbone)
        elif self.in_channels != 3:
            raise ValueError("ResNet50Classifier only supports 3 or 4 input channels")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, rgb=None, raw_like=None):
        x = rgb if self.in_channels == 3 else raw_like
        return self.backbone(x)

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        for name, p in self.named_parameters():
            if "backbone.fc" not in name:
                yield p

    def classifier_parameters(self) -> Iterable[nn.Parameter]:
        for name, p in self.named_parameters():
            if "backbone.fc" in name:
                yield p

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.encoder_parameters():
            p.requires_grad = bool(trainable)


class TwoStreamResNet50(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super(TwoStreamResNet50, self).__init__()
        self.rgb_encoder = _make_resnet50(pretrained=pretrained)
        self.raw_encoder = adapt_resnet_conv1_to_4ch(_make_resnet50(pretrained=pretrained))
        rgb_dim = self.rgb_encoder.fc.in_features
        raw_dim = self.raw_encoder.fc.in_features
        self.rgb_encoder.fc = Identity()
        self.raw_encoder.fc = Identity()
        self.classifier = nn.Sequential(
            nn.Linear(rgb_dim + raw_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb=None, raw_like=None):
        feature_rgb = self.rgb_encoder(rgb)
        feature_raw = self.raw_encoder(raw_like)
        feature = torch.cat([feature_rgb, feature_raw], dim=1)
        return self.classifier(feature)

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        for p in self.rgb_encoder.parameters():
            yield p
        for p in self.raw_encoder.parameters():
            yield p

    def classifier_parameters(self) -> Iterable[nn.Parameter]:
        for p in self.classifier.parameters():
            yield p

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.encoder_parameters():
            p.requires_grad = bool(trainable)


def build_model(name: str, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    if name == "rgb_only_resnet50":
        return ResNet50Classifier(in_channels=3, num_classes=num_classes, pretrained=pretrained)
    if name == "raw_only_resnet50":
        return ResNet50Classifier(in_channels=4, num_classes=num_classes, pretrained=pretrained)
    if name == "two_stream_resnet50":
        return TwoStreamResNet50(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    raise ValueError("Unknown model name: {}".format(name))

