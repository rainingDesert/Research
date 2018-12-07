import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
import numpy as np

import sys


class Simple_Classify_MIL(nn.Module):
    def __init__(self, num_classes=340):
        super(Simple_Classify_MIL, self).__init__()
        self.num_classes = num_classes
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return Y_prob, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, y_hat, Y):
        Y = Y.float()
        acc = y_hat.eq(Y).float().mean()

        return acc

    def calculate_objective(self, logits, Y):
        Y = Y.float()
        Y_prob = torch.clamp(logits, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (
                Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood


def get_simple_mil(**kwargs):
    model = Simple_Classify_MIL(**kwargs)

    return model

if __name__ == '__main__':
    print(get_simple_mil())
