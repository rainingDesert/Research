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
from functions.metrics import outline_to_large_area, get_max_binary_area

import sys


class Semi_VGG(nn.Module):

    def __init__(self, pre_trained_vgg, num_classes=200, inference=False, freeze_vgg=False):
        super(Semi_VGG, self).__init__()
        self.inference = inference

        self.features = pre_trained_vgg.features
        self.cls = self.classifier(512, num_classes)

        # self.linear_cls = nn.Sequential(
        #     nn.Linear(in_features=512 * 7 * 7, out_features=4096, bias=True),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features=4096, out_features=4096, bias=True),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        # )

        self.cls_to_1 = self.classifier(num_classes, 1)

        if freeze_vgg:
            for param in self.features.parameters():
                param.requires_grad = False

    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, out_planes, kernel_size=1, padding=0)  # fc8
        )

    # two branch
    def forward(self, x):
        # if self.inference:
        #     x.requires_grad_()
        #     x.retain_grad()
        #
        # base = self.features(x)  # （batch,512,7,7)
        # # class
        # logits = self.linear_cls(base.view(base.size(0), 512 * 7 * 7))
        #
        # # cam
        # avg_pool = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)  # （batch,512,7,7)
        # cam = self.cls(avg_pool)  # （batch,1,7,7)
        # norm_cam=self.normalize_atten_maps(cam)
        #
        # if self.inference:
        #     if x.grad is not None:
        #         x.grad.zero_()
        #
        #     cam.backward(gradient=cam, retain_graph=True)
        #     x_grad = torch.abs(x.grad)
        #
        #     return logits, cam, x_grad
        #
        # return logits, norm_cam

        # --------------------------------#

        if self.inference:
            x.requires_grad_()
            x.retain_grad()

        base = self.features(x)  # （batch,512,7,7)

        # cam
        avg_pool = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)  # （batch,512,7,7)
        cam = self.cls(avg_pool)  # （batch,200,7,7)

        one_cam = self.cls_to_1(cam)  # (batch,1,7,7)

        # norm_cam = self.normalize_atten_maps(one_cam)

        # class
        logits = torch.mean(torch.mean(cam, dim=2), dim=2)  # (1,200)

        if self.inference:
            if x.grad is not None:
                x.grad.zero_()

            cam.backward(gradient=cam, retain_graph=True)
            x_grad = torch.abs(x.grad)

            return logits, cam, x_grad

        return logits, one_cam

    def cal_element_loss(self, largest_norm_grad, cam):
        upsample_cam = F.upsample(cam, size=(224, 224), mode='bilinear', align_corners=True)

        cam_loss = torch.nn.BCELoss()
        sigmoid = nn.Sigmoid()
        return cam_loss(sigmoid(upsample_cam.view(upsample_cam.size(0), -1)),
                        largest_norm_grad.view(largest_norm_grad.size(0), -1))

    def norm_grad_2_binary(self, bi_x_grad):
        if len(bi_x_grad.size()) == 4:
            bi_x_grad = bi_x_grad.squeeze(1)

        grad_shape = bi_x_grad.size()
        outline = torch.zeros(grad_shape)
        for batch_idx in range(grad_shape[0]):
            thd = float(np.percentile(np.sort(bi_x_grad[batch_idx].view(-1).cpu().data.numpy()), 80))
            batch_outline = torch.zeros(bi_x_grad[batch_idx].size())
            high_pos = torch.gt(bi_x_grad[batch_idx], thd)
            batch_outline[high_pos.data] = 1.0
            large_batch_outline = outline_to_large_area(batch_outline)
            outline[batch_idx, :, :] = torch.from_numpy(large_batch_outline)

        return outline

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        # --------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def freeze(self, mode=True):
        if mode:
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            for param in self.features.parameters():
                param.requires_grad = True


def get_semi_vgg(pretrained=False, trained=False, **kwargs):
    pre_trained_model = models.vgg16(pretrained=pretrained)

    model = Semi_VGG(pre_trained_vgg=pre_trained_model, **kwargs)
    model.cuda()

    if trained:
        pre_trained_model = torch.load('../Save/model/semi_model_vgg_back.pt')
        pre_model_dict = pre_trained_model.state_dict()

        model_dict = model.state_dict()
        pre_trained_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
        model_dict.update(pre_trained_dict)
        model.load_state_dict(model_dict)

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
