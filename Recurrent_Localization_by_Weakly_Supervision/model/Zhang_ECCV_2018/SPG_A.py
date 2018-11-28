import torch
import torch.nn as nn


class SPP_A(nn.Module):
    def __init__(self, in_channels, rates=[1, 3, 6]):
        super(SPP_A, self).__init__()
        self.aspp = []
        for r in rates:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=128, kernel_size=3, dilation=r, padding=r),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, out_channels=128, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_conv1x1 = nn.Conv2d(128 * len(rates), 1, kernel_size=1)

    def forward(self, x):
        aspp_out = torch.cat([classifier(x) for classifier in self.aspp], dim=1)
        return self.out_conv1x1(aspp_out)


if __name__ == '__main__':
    print(SPP_A(in_channels=3))
