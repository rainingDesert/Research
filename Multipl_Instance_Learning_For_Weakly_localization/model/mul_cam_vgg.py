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

import sys


class Linear_vgg(nn.Module):
    def __init__(self, pre_trained_vgg, class_nums=200, inference=False, freeze_vgg=False):
        super(Linear_vgg, self).__init__()
        self.inference = inference
        self.freeze_vgg = freeze_vgg

        self.features = pre_trained_vgg.features
        self.cls1 = self.classifier(512, class_nums)
        self.cls2 = self.classifier(class_nums, class_nums)
        self.cls3 = self.classifier(class_nums, class_nums)

    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, out_planes, kernel_size=1, padding=0)  # fc8
        )

    def forward(self, x):
        if self.inference:
            x.requires_grad_()
            x.retain_grad()

        base = self.features(x)
        avg_pool = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)

        cam_1 = self.cls1(avg_pool)
        logits_1 = torch.mean(torch.mean(cam_1, dim=2), dim=2)
        peak_cam1, norm_cam1 = self.get_cutted_cam_peak(cam_1.detach())
        peak_cam1, norm_cam1 = peak_cam1.cuda(), norm_cam1.cuda()
        cut_cam_1 = cam_1 * peak_cam1

        cam_2 = self.cls2(cut_cam_1)
        logits_2 = torch.mean(torch.mean(cam_2, dim=2), dim=2)
        peak_cam2, norm_cam2 = self.get_cutted_cam_peak(cam_2.detach())
        peak_cam2, norm_cam2 = peak_cam2.cuda(), norm_cam2.cuda()
        cut_cam_2 = cam_2 * peak_cam2

        cam_3 = self.cls3(cut_cam_2)
        logits_3 = torch.mean(torch.mean(cam_3, dim=2), dim=2)

        return logits_1 + logits_2 + logits_3, [cam_1, cam_2, cam_3]

    def get_gradient_of(self, var, label):
        norm_var = self.normalize_atten_maps(var.detach())
        select_var = norm_var.new_empty(norm_var.size())
        select_var.zero_()

        for batch_idx in range(var.size(0)):
            select_var[batch_idx, label.data[batch_idx], :, :] = norm_var[batch_idx, label.data[batch_idx], :, :]

        output_grad = var.new_empty(var.size())
        output_grad.zero_()

        pos_high = torch.ge(select_var, 0.6)

        output_grad[pos_high.data] = 1.0

        return output_grad

    def get_cutted_cam_peak(self, cam):
        norm_cam = self.normalize_c_atten_maps(cam)
        peak_cam = torch.ones(norm_cam.size())

        high_pos = torch.ge(norm_cam, 0.005)
        peak_cam[high_pos.data] = 0.0

        return peak_cam, norm_cam

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        # --------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def normalize_c_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()
        atten_linear = atten_maps.view(atten_maps.size(0), atten_maps.size(1), -1)

        atten_exp = torch.exp(atten_linear)
        atten_norm = atten_exp / torch.sum(atten_exp, dim=1, keepdim=True)

        return atten_norm.view(atten_shape)

    def inference(self):
        self.inference = True


def get_mul_cam_vgg(pretrained=False, **kwargs):
    pre_trained_model = models.vgg16(pretrained=pretrained)

    model = Linear_vgg(pre_trained_vgg=pre_trained_model, **kwargs)
    return model


if __name__ == '__main__':
    print(get_linear_vgg(pretrained=True))

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
