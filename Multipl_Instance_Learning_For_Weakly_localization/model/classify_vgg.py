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

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

dilation = {
    'D1': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 1, 1, 1, 'N']
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=200, inference=False, freeze_vgg=False):
        super(VGG, self).__init__()
        self.inference = inference

        self.features = features
        self.cls = self.classifier(512, num_classes)

        if freeze_vgg:
            for param in self.features.parameters():
                param.requires_grad = False

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
        cam = self.cls(avg_pool)  # （batch,200,8,8)

        logits = torch.mean(torch.mean(cam, dim=2), dim=2)  # (1,200)

        if self.inference:
            label = torch.argmax(logits, dim=-1)
            output_grad = self.get_gradient_of(cam, label)
            if x.grad is not None:
                x.grad.zero_()

            cam.backward(gradient=output_grad, retain_graph=True)
            x_grad = torch.abs(x.grad)

        return logits, cam

    def get_gradient_of(self, var, label):
        norm_var = self.normalize_atten_maps(var.detach())
        select_var = norm_var.new_empty(norm_var.size())
        select_var.zero_()

        for batch_idx in range(var.size(0)):
            select_var[batch_idx, label.data[batch_idx], :, :] = norm_var[batch_idx, label.data[batch_idx], :, :]

        return select_var

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        # --------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def inference(self):
        self.inference = True


def make_layers(cfg, dilation=None, batch_norm=False):
    layers = []
    in_channels = 3
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def get_classify_vgg(pretrained=False, trained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = VGG(make_layers(cfg['D1'], dilation=dilation['D1']), **kwargs)
    if pretrained:
        pre_trained_dict = model_zoo.load_url(model_urls['vgg16'])
        model_dict = model.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
        model_dict.update(pre_trained_dict)
        model.load_state_dict(model_dict)

    if trained:
        pre_trained_model = torch.load('../Save/model/tradition_classify_model_vgg_back.pt')
        pre_model_dict = pre_trained_model.state_dict()

        model_dict = model.state_dict()
        pre_trained_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
        model_dict.update(pre_trained_dict)
        model.load_state_dict(model_dict)

    return model


if __name__ == '__main__':
    print(get_vgg(pretrained=True, num_classes=200))

'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
  )
  (cls): Sequential(
    (0): Dropout(p=0.5)
    (1): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): ReLU(inplace)
    (3): Dropout(p=0.5)
    (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): ReLU(inplace)
    (6): Conv2d(1024, 200, kernel_size=(1, 1), stride=(1, 1))
  )
)
'''
