from torchvision import models
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, finetune_all=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

        if not finetune_all:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.fc.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)