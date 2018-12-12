import torch.nn as nn
import torch
import numpy as np


class Basic_Model(nn.Module):
    def __init__(self, input_dims=2, sample_nums=10):
        super(Basic_Model, self).__init__()

        self.input_dims = input_dims
        self.sample_nums = sample_nums
        self.permutation_matrix = self.get_permutation_matrix()

        self.mlp_1 = nn.Sequential(
            nn.Linear(self.input_dims, 16),
            nn.ReLU(True),
            nn.Linear(16, 512),
            nn.ReLU(True)
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(512, 16),
            nn.ReLU(True),
            nn.Linear(16, 2),
            nn.ReLU(True)
        )

    def forward(self, x):
        mlp_1 = self.mlp_1(x)
        mlp_2 = self.mlp_2(mlp_1)

        return mlp_2

    def get_permutation_matrix(self):
        def swap_rows(m):
            new_m = np.vstack((m, m[0]))
            new_m = np.delete(new_m, (0), 0)

            return new_m

        trans = []
        for i in range(self.sample_nums):
            if len(trans) == 0:
                trans.append(np.identity(self.sample_nums))
            else:
                trans.append(swap_rows(trans[-1].copy()))

        return torch.from_numpy(np.array(trans)).type(torch.cuda.FloatTensor).view(self.sample_nums ** 2,
                                                                                   self.sample_nums)

    def cal_distance(self, m1, m2):
        permutated = torch.mm(self.permutation_matrix, m2)
        matrix1_3d = m1.unsqueeze(0).repeat(self.sample_nums, 1, 1).view(-1, self.input_dims)
        result = torch.sum((matrix1_3d - permutated) ** 2, dim=-1) ** 0.5

        result_min, _ = torch.min(result.view(self.sample_nums, self.sample_nums).transpose(1, 0), dim=-1)
        return torch.sum(result_min)

    def cal_distance_loss(self, logits, b_y):
        loss = torch.tensor([0.0]).type(torch.cuda.FloatTensor)
        for batch_idx in range(logits.size()[0]):
            sum_1 = self.cal_distance(logits[batch_idx], b_y[batch_idx])
            sum_2 = self.cal_distance(b_y[batch_idx], logits[batch_idx])
            loss += (sum_1 + sum_2) / (self.sample_nums * 2)

        return loss
