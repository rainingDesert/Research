import torch
import torch.nn as nn


class SPP_B(nn.Module):
    def __init__(self, in_channels, num_classes=1000, rates=[1, 3, 6]):
        super(SPP_B, self).__init__()
        self.aspp = []
        for r in rates:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=1024, kernel_size=3, dilation=r, padding=r),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1024, out_channels=1024, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_conv1x1 = nn.Conv2d(1024, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        aspp_out = torch.mean([classifier(x) for classifier in self.aspp], dim=1)
        return self.out_conv1x1(aspp_out)


if __name__ == '__main__':
    print(SPP_B(in_channels=3))
