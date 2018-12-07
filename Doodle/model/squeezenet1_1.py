import torch.nn as nn
from torchvision import models
import torch


class squeezenet(nn.Module):
    def __init__(self, model, num_classes):
        super(squeezenet, self).__init__()

        # feature encoding
        self.features = model.features

        self.features[0].apply(self.squeeze_weights)

        self.cls = model.classifier
        self.cls[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

        print()

    def forward(self, x):
        x = self.features(x)  # (bn,512,27,27)
        x = self.cls(x)

        return torch.squeeze(x)

    @staticmethod
    def squeeze_weights(m):
        m.weight.data = m.weight.data.sum(dim=1)[:, None]
        m.in_channels = 1

    def linear_classifier(self, in_planes, out_planes):
        return nn.Linear(in_planes, out_planes)


def _squeezenet(num_classes=340, pre_trained=True):
    model = squeezenet(models.squeezenet1_1(pre_trained), num_classes)

    return model


if __name__ == '__main__':
    test = _squeezenet()
    print(test(torch.randn(12, 1, 224, 224)).size())

'''
squeezenet(
  (model): SqueezeNet(
    (features): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2))
      (1): ReLU(inplace)
      (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (3): Fire(
        (squeeze): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace)
        (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace)
        (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace)
      )
      (4): Fire(
        (squeeze): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace)
        (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace)
        (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace)
      )
      (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (6): Fire(
        (squeeze): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace)
        (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace)
        (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace)
      )
      (7): Fire(
        (squeeze): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace)
        (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace)
        (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace)
      )
      (8): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
      (9): Fire(
        (squeeze): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace)
        (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace)
        (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace)
      )
      (10): Fire(
        (squeeze): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace)
        (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace)
        (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace)
      )
      (11): Fire(
        (squeeze): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace)
        (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace)
        (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace)
      )
      (12): Fire(
        (squeeze): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        (squeeze_activation): ReLU(inplace)
        (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        (expand1x1_activation): ReLU(inplace)
        (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (expand3x3_activation): ReLU(inplace)
      )
    )
    (classifier): Sequential(
      (0): Dropout(p=0.5)
      (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
      (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
    )
  )
)
'''
