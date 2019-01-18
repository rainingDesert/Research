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

        self.features = nn.Sequential(*pre_trained_vgg.features[:-1])
        self.cls = nn.Sequential(
            nn.Conv2d(512, self.class_nums, kernel_size=1, padding=0)  # fc8
        )
        self.fc = nn.Sequential(
            nn.Linear(200, 200 * 14 * 14)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=200, out_channels=512, kernel_size=3, stride=2, padding=0),
            nn.ConvTranspose2d(in_channels=512, out_channels=200, kernel_size=3, stride=2, padding=0)
        )

        if self.freeze_vgg:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.inference:
            x.requires_grad_()
            x.retain_grad()

        base = self.features(x)
        cam = self.cls(base)
        logits = torch.mean(torch.mean(cam, dim=2), dim=2)
        fc = self.fc(logits)
        fc_3d = fc.view(-1, 200, 14, 14)

        deconv_cam = self.deconv(fc_3d)
        logits2 = torch.mean(torch.mean(deconv_cam, dim=2), dim=2)

        if self.inference:
            pass

        return logits, logits2, cam, deconv_cam

    def norm_cam_2_binary(self, bi_x_grad):
        thd = float(np.percentile(np.sort(bi_x_grad.view(-1).cpu().data.numpy()), 80))
        outline = torch.zeros(bi_x_grad.size())
        high_pos = torch.gt(bi_x_grad, thd)
        outline[high_pos.data] = 1.0

        return outline


def get_vgg_deconv2_model(pretrained=True, **kwargs):
    pre_trained_model = models.vgg16(pretrained=pretrained)

    model = Deconv_vgg2(pre_trained_vgg=pre_trained_model, **kwargs)
    model.cuda()

    return model
