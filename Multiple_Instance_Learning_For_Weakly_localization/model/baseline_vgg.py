import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
import numpy as np
import torchvision.models as models
from functions.metrics import outline_to_large_area

import sys


class Baseline_vgg(nn.Module):
    def __init__(self, pre_trained_vgg, class_nums=200, inference=False, freeze_vgg=False):
        super(Baseline_vgg, self).__init__()
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
            print()

        return logits, cam

    def norm_grad_2_binary(self, bi_x_grad):
        if len(bi_x_grad.size()) == 4:
            bi_x_grad = bi_x_grad.squeeze(1)

        grad_shape = bi_x_grad.size()
        outline = torch.zeros(grad_shape)
        for batch_idx in range(grad_shape[0]):
            thd = float(np.percentile(np.sort(bi_x_grad[batch_idx].view(-1).cpu().data.numpy()), 90))
            batch_outline = torch.zeros(bi_x_grad[batch_idx].size())
            high_pos = torch.gt(bi_x_grad[batch_idx], thd)
            batch_outline[high_pos.data] = 1.0
            large_batch_outline = outline_to_large_area(batch_outline)
            outline[batch_idx, :, :] = torch.from_numpy(large_batch_outline)

        return outline


def get_baseline_vgg(pretrained=False, trained=False, **kwargs):
    pre_trained_model = models.vgg16(pretrained=pretrained)

    model = Baseline_vgg(pre_trained_vgg=pre_trained_model, **kwargs)
    model.cuda()

    if trained:
        model.load_state_dict(torch.load('../Save/model/baseline_vgg.pt'))

    return model
