from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import scipy.stats as ss
import os
import numpy as np
import pandas as pd


class Function_Generator_Dataset(Dataset):
    def __init__(self, x_nums=10, y_nums=100, iter_nums=1000):
        self.x_nums = x_nums
        self.y_nums = y_nums
        self.iter_nums = iter_nums

    def __getitem__(self, index):
        random_distribution_df = pd.DataFrame.from_dict(self.gen_random_gaussian())

        x_df = random_distribution_df.sample(n=self.x_nums)
        y_df = random_distribution_df.drop(x_df.index).sample(n=self.y_nums)

        x = np.array(x_df)
        y = np.array(y_df)

        # self.plot_points(random_distribution_df)

        return torch.from_numpy(x).type(torch.FloatTensor), torch.from_numpy(y).type(torch.FloatTensor)

    def __len__(self):
        return self.iter_nums

    def gen_random_gaussian(self, nums=1000, x_mu=[-0.35, 0.35], x_sigma=[0, 0.25], y_mu=[-0.35, 0.35],
                            y_sigma=[0, 0.25]):
        x = np.random.normal(np.random.uniform(x_mu[0], x_mu[1], size=1),
                             scale=np.random.uniform(x_sigma[0], x_sigma[1], size=1), size=nums)
        y = np.random.normal(np.random.uniform(y_mu[0], y_mu[1], size=1),
                             scale=np.random.uniform(y_sigma[0], y_sigma[1], size=1), size=nums)

        return {'x': x, 'y': y}

    def load_funcdata_from_functions(self):
        func_data = {}
        func_index = 0
        for func in self.functions:
            func_data[func.__name__ + '.' + str(func_index)] = func()
            func_index += 1

        return func_data

    def plot_points(self, random_distribution):
        np_points = np.array(random_distribution)
        plt.scatter(x=np_points[:, 0], y=np_points[:, 1], s=20)
        plt.show()


# get all distribution functions
def get_functions(*args):
    return [arg for arg in args]


# y=ax+b
def linear_function(a=1, b=0, noise=True, x_range=[-500, 500], nums=10):
    def linear_function_exc():
        x = torch.unsqueeze(torch.from_numpy(np.random.randint(x_range[0], x_range[1], nums)).type(torch.FloatTensor),
                            dim=1)

        y = a * x + b
        if noise:
            y += random.random() * torch.rand(x.size()) * x

        return [x, y]

    return linear_function_exc


# y=a(x+c)^2+b
def quadratic_function(a=1, b=0, c=0, noise=True, x_range=[-500, 500], nums=10):
    def quadratic_function_exc():
        x = np.random.uniform(x_range[0], x_range[1], nums)

        y = a * (x + c) ** 2 + b
        if noise:
            y += np.random.normal(0, 1, y.shape)

        return [x, y]

    return quadratic_function_exc


# y=a*log(x+c)+b
def log_function(a=1, b=0, c=0, noise=True, x_range=[0, 500], nums=10):
    if x_range[0] > c:
        x_range[0] = -c

    def log_function_exc():
        x = torch.unsqueeze(torch.from_numpy(np.random.randint(x_range[0], x_range[1], nums)).type(torch.FloatTensor),
                            dim=1)

        y = a * torch.log(x + c) + b
        if noise:
            y += random.random() * torch.rand(x.size()) * 2

        return [x, y]

    return log_function_exc


def gaussian(x_mu=0, x_sigma=2, y_mu=0, y_sigma=5, noise=True, nums=100):
    def gaussian_exc():
        x = np.random.normal(x_mu, x_sigma, nums)
        y = np.random.normal(y_mu, y_sigma, nums)

        return [x, y]

    return gaussian_exc
