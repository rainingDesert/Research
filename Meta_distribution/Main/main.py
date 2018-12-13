import sys

sys.path.append('../')

from Dataset.Function_Generator_Dataset import Function_Generator_Dataset, get_functions, quadratic_function, gaussian
import argparse
from torch.utils.data import DataLoader
from models.basic_model import Basic_Model
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    return args


def train():
    functions_train = get_functions(
        # quadratic_function(nums=1000, x_range=[-10, 10], b=300),
        # quadratic_function(nums=1000, x_range=[-10, 10], b=-120),
        # quadratic_function(nums=1000, x_range=[20, 40], b=10, c=-30),
        # quadratic_function(nums=1000, x_range=[-20, 0], c=10, b=-10),
        # quadratic_function(nums=1000, x_range=[180, 200], c=-190, b=-350),
        # gaussian(nums=1000, y_mu=200, x_mu=100, x_sigma=5, y_sigma=0.1),
        # gaussian(nums=1000, y_mu=20, x_mu=60, x_sigma=5, y_sigma=0.1),
        # gaussian(nums=1000, y_mu=70, x_mu=-30, x_sigma=5, y_sigma=0.1),
        # gaussian(nums=1000, y_mu=-200, x_mu=46, x_sigma=5, y_sigma=0.1),
        # gaussian(nums=1000, y_mu=33, x_mu=89, x_sigma=5, y_sigma=0.1)
        gaussian(nums=1000, y_mu=8, x_mu=10, x_sigma=1, y_sigma=1),
        gaussian(nums=1000, y_mu=15, x_mu=5, x_sigma=1, y_sigma=1),
        gaussian(nums=1000, y_mu=23, x_mu=2, x_sigma=1, y_sigma=1),
        gaussian(nums=1000, y_mu=19, x_mu=26, x_sigma=1, y_sigma=1),
        gaussian(nums=1000, y_mu=5, x_mu=13, x_sigma=1, y_sigma=1)
    )

    dataset_train = Function_Generator_Dataset(functions=functions_train, mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    functions_test = get_functions(
        # # quadratic_function(nums=1000, x_range=[-5, 5], b=300),
        # # quadratic_function(a=-1, nums=1000, x_range=[-5, 5], b=-120),
        # # quadratic_function(a=2, nums=1000, x_range=[25, 37], b=10, c=-20),
        # # quadratic_function(nums=1000, x_range=[12, 17], c=-12, b=-10),
        # # quadratic_function(nums=1000, x_range=[0, 20], c=-10, b=-350),
        # gaussian(nums=1000, y_mu=15, x_mu=15, x_sigma=1, y_sigma=1),
        # gaussian(nums=1000, y_mu=15, x_mu=15, x_sigma=1, y_sigma=1),
        # gaussian(nums=1000, y_mu=15, x_mu=15, x_sigma=1, y_sigma=1),
        # gaussian(nums=1000, y_mu=15, x_mu=15, x_sigma=1, y_sigma=1),
        # gaussian(nums=1000, y_mu=15, x_mu=15, x_sigma=1, y_sigma=1)

        quadratic_function(nums=1000, x_range=[-60, -80], c=70, b=20),
        gaussian(nums=1000, y_mu=10, x_mu=20, x_sigma=5, y_sigma=0.1)
    )

    dataset_test = Function_Generator_Dataset(functions=functions_train, mode='train')
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    model = Basic_Model()
    model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        for step, (b_x, b_y) in enumerate(dataloader_train):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            random_index = torch.randint(0, 100, (10,)).long()
            b_x = b_x[:, random_index, :]
            b_x = b_x.view(b_x.size(0), -1)

            logits = model(b_x)
            loss = model.cal_distance_loss(logits, b_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 500 == 0:
                print('train loss: {}'.format(str(loss.item())))
                for _, (b_x, b_y) in enumerate(dataloader_test):
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                    random_index = torch.randint(0, 100, (10,)).long()
                    b_x = b_x[:, random_index, :]
                    b_x = b_x.view(b_x.size(0), -1)
                    logits = model(b_x)

                    model.plot_reuslts_points_and_gt(logits.cpu().data.numpy(), b_x.unsqueeze(1).cpu().data.numpy(),
                                                     b_y.cpu().data.numpy())

                    print('fake_mean: {}'.format(str(torch.mean(logits[:, :, 1]))))
                    print('real_mean: {}'.format(str(torch.mean(b_y[:, :, 1]))))
                    break


if __name__ == '__main__':
    args = parse_args()
    train()
