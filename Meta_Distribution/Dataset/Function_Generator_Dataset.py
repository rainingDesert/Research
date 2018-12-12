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
    def __init__(self, functions, mode='train', train_data_size=0.8):
        self.mode = mode
        self.train_data_size = train_data_size

        self.functions = functions
        self.func_data = self.load_funcdata_from_functions()
        self.category = list(self.func_data.keys())

        # get data
        self.data = []
        for key in self.func_data.keys():
            for i in range(len(self.func_data[key][0])):
                self.data.append([self.func_data[key][0][i][0], self.func_data[key][1][i][0], key])
        self.data = pd.DataFrame(self.data, columns=['x', 'y', 'label']).sample(frac=1)
        self.train_data = self.data.sample(frac=self.train_data_size)
        self.test_data = self.data.drop(self.train_data.index)

        self.train_data.reset_index(inplace=True, drop=True)
        self.test_data.reset_index(inplace=True, drop=True)

    def __getitem__(self, index):
        if self.mode == 'train':
            random_row = self.train_data.sample(n=1)
            target_class_data = self.train_data[self.train_data['label'] == random_row['label'].values[0]]

            x_df = target_class_data.sample(n=10)
            y_df = target_class_data.drop(x_df.index).sample(n=10)

            x = np.array(x_df[['x', 'y']])
            y = np.array(y_df[['x', 'y']])

            return torch.from_numpy(x).type(torch.FloatTensor), torch.from_numpy(y).type(torch.FloatTensor)

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'test':
            return len(self.test_data)
        else:
            return self.val_data

    def load_funcdata_from_functions(self):
        func_data = {}
        func_index = 0
        for func in self.functions:
            func_data[func.__name__ + '.' + str(func_index)] = func()
            func_index += 1

        return func_data

    def plot_all_points(self, data_path='../Save/all_function_points.png'):
        for key in self.func_data.keys():
            color = '#' + "".join(random.sample("0123456789abcdef", 6))
            plt.scatter(x=self.func_data[key][0], y=self.func_data[key][1], color=color, s=20,
                        label=key)

        plt.legend()
        plt.savefig(data_path)


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
        x = np.expand_dims(np.random.normal(x_mu, x_sigma, nums), 1)
        y = np.expand_dims(np.random.normal(y_mu, y_sigma, nums), 1)

        return [x, y]

    return gaussian_exc
