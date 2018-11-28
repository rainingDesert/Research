import torch.nn as nn
from torchvision import models
import torch


class Linear_Transform(nn.Module):
    def __init__(self, num_classes=20):
        super(Linear_Transform, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.Dropout(0.5)
        )

    def forward(self, *input):
        x = self.linear(*input)

        return x


def function_linear_transform():
    model = Linear_Transform()

    return model
