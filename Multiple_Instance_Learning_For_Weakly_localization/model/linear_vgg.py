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


class Linear_vgg(nn.Module):
    def __init__(self, pre_trained_vgg, class_nums=200, inference=False, freeze_vgg=True):
        super(Linear_vgg, self).__init__()
        self.inference = inference
        self.freeze_vgg = freeze_vgg

        self.features = pre_trained_vgg.features
        self.cls1 = nn.Sequential(
            pre_trained_vgg.classifier[0],
            pre_trained_vgg.classifier[1],
        )

        self.cls2 = nn.Sequential(
            pre_trained_vgg.classifier[2],
            pre_trained_vgg.classifier[3],
            pre_trained_vgg.classifier[4],
        )

        self.cls3 = nn.Sequential(
            pre_trained_vgg.classifier[5],
            nn.Linear(4096, class_nums, bias=True)
        )

    def forward(self, x):
        if self.inference:
            x.requires_grad_()
            x.retain_grad()

        base = self.features(x)
        linear_base = base.view(base.size(0), -1)

        relu1 = self.cls1(linear_base)
        relu2 = self.cls2(relu1)
        final = self.cls3(relu2)

        if self.inference:
            if x.grad is not None:
                x.grad.zero_()

            relu2.backward(gradient=relu2, retain_graph=True)
            init_x_grad = torch.abs(x.grad)
            init_norm_x_grad = self.norm_grad_2_binary(init_x_grad)

            return init_norm_x_grad

    def inference(self):
        self.inference = True

    def norm_grad_2_binary(self, bi_x_grad):
        sum_bi_x_grad = bi_x_grad.sum(1)  # (batch,256,256)
        grad_shape = sum_bi_x_grad.size()
        outline = torch.zeros(grad_shape)
        for batch_idx in range(grad_shape[0]):
            thd = float(np.percentile(np.sort(sum_bi_x_grad[batch_idx].view(-1).cpu().data.numpy()), 75))
            batch_outline = torch.zeros(sum_bi_x_grad[batch_idx].size())
            high_pos = torch.gt(sum_bi_x_grad[batch_idx], thd)
            batch_outline[high_pos.data] = 1.0
            large_batch_outline = outline_to_large_area(batch_outline)
            outline[batch_idx, :, :] = torch.from_numpy(large_batch_outline)

        return outline


def get_linear_vgg(pretrained=False, **kwargs):
    pre_trained_model = models.vgg16(pretrained=pretrained)

    model = Linear_vgg(pre_trained_vgg=pre_trained_model, **kwargs)
    return model


if __name__ == '__main__':
    print()

'''
VGG(
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
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''
