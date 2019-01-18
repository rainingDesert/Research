import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

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


class Dilation(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Dilation, self).__init__()
        self.d1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=1),
            nn.ReLU(True)
        )

        self.d3 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), dilation=3),
            nn.ReLU(True)
        )

        self.d5 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(5, 5), dilation=5),
            nn.ReLU(True)
        )

        self.d7 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(7, 7), dilation=7),
            nn.ReLU(True)
        )

    def forward(self, x):
        d1 = self.d1(x)
        d3 = self.d3(x)
        d5 = self.d5(x)
        d7 = self.d7(x)
        return d1 + (d3 + d5 + d7) / 3


class Dilation_vgg(nn.Module):
    def __init__(self, features, args, inference=False, freeze_vgg=False):
        super(Dilation_vgg, self).__init__()
        self.inference = inference
        self.freeze_vgg = freeze_vgg
        self.class_nums = args.class_nums

        self.features = features.features
        self.cls = self.classifier(512, self.class_nums)

        self.block1 = nn.Sequential(
            self.features[0],
            self.features[1],
            self.features[2],
            self.features[3],
            self.features[4]
        )

        self.block2 = nn.Sequential(
            self.features[5],
            self.features[6]
        )
        self.block2_dilation=Dilation(128,128)

        self.block3 = nn.Sequential(
            self.features[10],
            self.features[11],
            self.features[12],
            self.features[13],
        )
        self.block3_dilation = Dilation(256, 256)

        self.block4 = nn.Sequential(
            self.features[17],
            self.features[18],
            self.features[19],
            self.features[20]
        )
        self.block4_dilation = Dilation(512, 512)

        self.block5 = nn.Sequential(
            self.features[24],
            self.features[25],
            self.features[26],
            self.features[27]
        )
        self.block5_dilation = Dilation(512, 512)

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

        block1 = self.block1(x)

        block2 = self.block2(block1)
        block2 = self.block2_dilation(block2)
        block2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(block2) #(56,56)

        block3 = self.block3(block2)
        block3 = self.block3_dilation(block3)
        block3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(block3)

        block4 = self.block4(block3)
        block4 = self.block4_dilation(block4)
        block4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(block4)

        block5 = self.block5(block4)
        block5 = self.block5_dilation(block5)
        block5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(block5)

        base = F.avg_pool2d(block5, kernel_size=3, stride=1, padding=1)
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


def get_dilation_vgg_model(**kwargs):
    # features = make_layers(cfg['D1'], dilation=dilation['D1'])
    #
    # model = Dilation_vgg(pre_trained_vgg=features, **kwargs)
    #
    # pretrained_dict = {k: v for k, v in model_zoo.load_url(model_urls['vgg16']).items() if 'features' in k}
    # model_dict = model.state_dict()
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # model.cuda()

    pre_trained_model = models.vgg16(pretrained=True)

    model = Dilation_vgg(features=pre_trained_model, **kwargs)
    model.cuda()

    return model
