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


class Every_cam_layer_vgg(nn.Module):
    def __init__(self, pre_trained_vgg, class_nums=20, inference=False, freeze_vgg=False):
        super(Every_cam_layer_vgg, self).__init__()
        self.inference = inference
        self.freeze_vgg = freeze_vgg

        self.features = pre_trained_vgg.features
        self.cls = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            # nn.Dropout(0.5),
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

    model = Every_cam_layer_vgg(pre_trained_vgg=pre_trained_model, **kwargs)
    model.cuda()

    if trained:
        model.load_state_dict(torch.load('../save/models/binary_vgg.pt'))

    return model


if __name__ == '__main__':
    model = get_binary_vgg()
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            print(module)

'''
Every_cam_layer_vgg(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (cls): Sequential(
    (0): Dropout(p=0.5)
    (1): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): ReLU(inplace)
    (3): Dropout(p=0.5)
    (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): ReLU(inplace)
    (6): Conv2d(1024, 20, kernel_size=(1, 1), stride=(1, 1))
  )
)
'''
