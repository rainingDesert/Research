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

        self.features = nn.Sequential(*pre_trained_vgg.features[:-7])
<<<<<<< HEAD
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
=======
        self.cls = self.classifier(512, self.class_nums)
>>>>>>> 3e238dfe5310c9a9caa9a9ec4a15b22b8a2a44ab

        if self.freeze_vgg:
            for param in self.features.parameters():
                param.requires_grad = False

<<<<<<< HEAD
=======
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

>>>>>>> 3e238dfe5310c9a9caa9a9ec4a15b22b8a2a44ab
    def forward(self, x):
        if self.inference:
            x.requires_grad_()
            x.retain_grad()

        base = self.features(x)
<<<<<<< HEAD
        cam = self.cls(base)
        logits = torch.mean(torch.mean(cam, dim=2), dim=2)
        fc = self.fc(logits)
        fc_3d = fc.view(-1, 200, 14, 14)

        deconv_cam = self.deconv(fc_3d)
        logits2 = torch.mean(torch.mean(deconv_cam, dim=2), dim=2)
=======
        base = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        cam = self.cls(base)

        logits = torch.mean(torch.mean(cam, dim=2), dim=2)
>>>>>>> 3e238dfe5310c9a9caa9a9ec4a15b22b8a2a44ab

        if self.inference:
            pass

<<<<<<< HEAD
        return logits, logits2, cam, deconv_cam
=======
        return logits, cam
>>>>>>> 3e238dfe5310c9a9caa9a9ec4a15b22b8a2a44ab

    def norm_cam_2_binary(self, bi_x_grad):
        thd = float(np.percentile(np.sort(bi_x_grad.view(-1).cpu().data.numpy()), 80))
        outline = torch.zeros(bi_x_grad.size())
        high_pos = torch.gt(bi_x_grad, thd)
        outline[high_pos.data] = 1.0

        return outline


<<<<<<< HEAD
def get_vgg_deconv_cgf_model(pretrained=True, **kwargs):
    pre_trained_model = models.vgg16(pretrained=pretrained)

    model = Deconv_vgg2(pre_trained_vgg=pre_trained_model, **kwargs)
    model.cuda()

    return model
=======

def get_vgg_deconv_cgf_model(pretrained=True, **kwargs):
    pre_trained_model = models.vgg16(pretrained=pretrained)
    # print(pre_trained_model)
    model = Deconv_vgg2(pre_trained_vgg=pre_trained_model, **kwargs)
    # print(model)
    # model.cuda()

    return model
>>>>>>> 3e238dfe5310c9a9caa9a9ec4a15b22b8a2a44ab
