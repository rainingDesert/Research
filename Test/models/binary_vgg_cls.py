import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
import torchvision.models as models

import sys


class Binary_vgg(nn.Module):
    def __init__(self, pre_trained_vgg, class_nums=2, inference=False, freeze_vgg=False):
        super(Binary_vgg, self).__init__()
        self.inference = inference
        self.freeze_vgg = freeze_vgg

        self.features = pre_trained_vgg.features
        self.cls = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, class_nums, kernel_size=1, padding=0)  # fc8
        )

        if self.freeze_vgg:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.inference:
            x.requires_grad_()
            x.retain_grad()

        base = self.features(x)
        avg_pool = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        cam = self.cls(avg_pool)
        logits = torch.mean(torch.mean(cam, dim=2), dim=2)  # (1,200)

        if self.inference:
            pass

        return logits, cam


def get_binary_vgg(pretrained=False, trained=False, **kwargs):
    pre_trained_model = models.vgg16(pretrained=pretrained)

    model = Binary_vgg(pre_trained_vgg=pre_trained_model, **kwargs)
    model.cuda()

    if trained:
        model.load_state_dict(torch.load('../save/models/binary_vgg.pt'))

    return model
