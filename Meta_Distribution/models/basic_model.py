import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt




class Basic_Model(nn.Module):
    def __init__(self, input_dims=2 * 10, sample_nums=100, noise_dim=2, feat_dim=2):
        super(Basic_Model, self).__init__()

        self.input_dims = input_dims
        self.sample_nums = sample_nums
        self.noise_dim = noise_dim
        self.feat_dim = feat_dim
        self.permutation_matrix = self.get_permutation_matrix()
        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=2,
            hidden_size=32,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,
            # input & output will have batch size as first dimension. e.g. (batch, time_step, input_size)
        )

        self.mlp_1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 4),
            nn.Tanh(),

        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(4 + self.noise_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        #print(x.size())
        x, _ = self.rnn(x, None)  # None means no initial hidden state
        x = x[:, -1, :]


        x = self.mlp_1(x)
        x = torch.unsqueeze(x,dim=1)

        noise = torch.rand(x.size(0), self.sample_nums, self.noise_dim).cuda()
        x = x.repeat((1,self.sample_nums, 1))

        x = torch.cat([x, noise], dim=2)
        x = self.mlp_2(x)

        return x

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
        matrix1_3d = m1.unsqueeze(0).repeat(self.sample_nums, 1, 1).view(-1, self.feat_dim)
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

    def plot_reuslts_points_and_gt(self, logits, b_x, b_y):
        for batch_idx in range(logits.shape[0]):
            reshape_bx = b_x[batch_idx].reshape(-1, b_y.shape[-1])
            plt.scatter(x=reshape_bx[:, 0], y=reshape_bx[:, 1],
                        color='b', s=20, label='b_x')
            plt.scatter(x=logits[batch_idx][:, 0], y=logits[batch_idx][:, 1],
                        color='r', s=20, label='logits')
            plt.scatter(x=b_y[batch_idx][:, 0], y=b_y[batch_idx][:, 1],
                        color='g', s=20, label='b_y')
        plt.ion()
        plt.legend()
        plt.savefig('../Save/images/result_x_y.png')
        plt.show()
        plt.close()

'''
if __name__ == '__main__':
    m1 = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).type(torch.cuda.FloatTensor)
    m2 = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [2, 0, 1]])).type(torch.cuda.FloatTensor)

    model = Basic_Model(sample_nums=3,feat_dim=3)
    model.cuda()

    loss = model.cal_distance(m1, m2)
    loss=model.cal_distance(m2, m1)
    print()
'''