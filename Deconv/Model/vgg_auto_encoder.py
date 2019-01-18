import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models


class Vgg_auto_encoder(nn.Module):
    def __init__(self, pre_trained_vgg, args, inference=False, freeze_vgg=False):
        super(Vgg_auto_encoder, self).__init__()
        self.inference = inference
        self.freeze_vgg = freeze_vgg
        self.class_nums = args.class_nums

        self.features1 = nn.Sequential(
            pre_trained_vgg.features[0],
            nn.BatchNorm2d(64),
            pre_trained_vgg.features[1],
            pre_trained_vgg.features[2],
            nn.BatchNorm2d(64),
            pre_trained_vgg.features[3]
        )
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.features2 = nn.Sequential(
            pre_trained_vgg.features[5],
            nn.BatchNorm2d(128),
            pre_trained_vgg.features[6],
            pre_trained_vgg.features[7],
            nn.BatchNorm2d(128),
            pre_trained_vgg.features[8]
        )
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.features3 = nn.Sequential(
            pre_trained_vgg.features[10],
            nn.BatchNorm2d(256),
            pre_trained_vgg.features[11],
            pre_trained_vgg.features[12],
            nn.BatchNorm2d(256),
            pre_trained_vgg.features[13],
            pre_trained_vgg.features[14],
            nn.BatchNorm2d(256),
            pre_trained_vgg.features[15]
        )
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.features4 = nn.Sequential(
            pre_trained_vgg.features[17],
            nn.BatchNorm2d(512),
            pre_trained_vgg.features[18],
            pre_trained_vgg.features[19],
            nn.BatchNorm2d(512),
            pre_trained_vgg.features[20],
            pre_trained_vgg.features[21],
            nn.BatchNorm2d(512),
            pre_trained_vgg.features[22]
        )
        self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.features5 = nn.Sequential(
            pre_trained_vgg.features[24],
            nn.BatchNorm2d(512),
            pre_trained_vgg.features[25],
            pre_trained_vgg.features[26],
            nn.BatchNorm2d(512),
            pre_trained_vgg.features[27],
            pre_trained_vgg.features[28],
            nn.BatchNorm2d(512),
            pre_trained_vgg.features[29]
        )
        self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.cls_before=nn.Sequential(
            nn.Conv2d(512, self.class_nums, kernel_size=1, padding=0)  # fc8
        )

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)  # (14,14)
        self.deconv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            # nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            # nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            # nn.ReLU(True),
        )
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)  # (28,28)
        self.deconv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            # nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            # nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),
        )

        self.cls = nn.Sequential(
            nn.Conv2d(256, self.class_nums, kernel_size=1, padding=0)  # fc8
        )

        if self.freeze_vgg:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.inference:
            x.requires_grad_()
            x.retain_grad()

        # base = self.features(x)
        # deconv=self.deconv(base)
        # cam = self.cls(deconv)
        # logits = torch.mean(torch.mean(cam, dim=2), dim=2)
        #
        # if self.inference:
        #     pass

        output1, indices1 = self.pool1(self.features1(x))
        output2, indices2 = self.pool2(self.features2(output1))
        output3, indices3 = self.pool3(self.features3(output2))
        output4, indices4 = self.pool4(self.features4(output3))
        output5, indices5 = self.pool5(self.features5(output4))

        encode_cam=self.cls_before(output5)
        encode_logits=torch.mean(torch.mean(encode_cam, dim=2), dim=2)

        unpool1 = self.unpool1(output5, indices5, output_size=output4.size())
        deconv1 = self.deconv1(unpool1)
        unpool2 = self.unpool2(deconv1, indices4, output_size=output3.size())
        deconv2 = self.deconv2(unpool2)

        cam = self.cls(deconv2)
        logits = torch.mean(torch.mean(cam, dim=2), dim=2)

        return logits*encode_logits, cam

    def norm_cam_2_binary(self, bi_x_grad):
        thd = float(np.percentile(np.sort(bi_x_grad.view(-1).cpu().data.numpy()), 80))
        outline = torch.zeros(bi_x_grad.size())
        high_pos = torch.gt(bi_x_grad, thd)
        outline[high_pos.data] = 1.0

        return outline


def get_vgg_auto_encoder_model(pretrained=True, **kwargs):
    pre_trained_model = models.vgg16(pretrained=pretrained)

    model = Vgg_auto_encoder(pre_trained_vgg=pre_trained_model, **kwargs)
    model.cuda()

    return model
