import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models


class Deconv_vgg2(nn.Module):
    def __init__(self, pre_trained_vgg, args, inference=False, freeze_vgg=False):
        super(Deconv_vgg2, self).__init__()
        self.inference = inference
        self.freeze_vgg = freeze_vgg
        self.class_nums = args.class_nums

        self.features = nn.Sequential(*pre_trained_vgg.features[:10])
        self.cls = self.classifier(128, self.class_nums)

        if self.freeze_vgg:
            for param in self.features.parameters():
                param.requires_grad = False

    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
            # nn.Dropout(0.5),
            nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, out_planes, kernel_size=1, padding=0)  # fc8
        )

    def forward(self, x):
        if self.inference:
            x.requires_grad_()
            x.retain_grad()

        base = self.features(x)
        base = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        cam = self.cls(base)

        logits = torch.mean(torch.mean(cam, dim=2), dim=2)

        if self.inference:
            pass

        return logits, cam

    def norm_cam_2_binary(self, bi_x_grad):
        thd = float(np.percentile(np.sort(bi_x_grad.view(-1).cpu().data.numpy()), 80))
        outline = torch.zeros(bi_x_grad.size())
        high_pos = torch.gt(bi_x_grad, thd)
        outline[high_pos.data] = 1.0

        return outline


def get_vgg_deconv_cgf_model(pretrained=True, **kwargs):
    pre_trained_model = models.vgg16(pretrained=pretrained)
    model = Deconv_vgg2(pre_trained_vgg=pre_trained_model, **kwargs)
    model.cuda()

    return model
