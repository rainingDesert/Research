import sys

sys.path.append('../')

from Dataset.Function_Generator_Dataset import Function_Generator_Dataset, get_functions, quadratic_function, gaussian
import argparse
from torch.utils.data import DataLoader
from models.basic_model import Basic_Model
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    return args


def train():
    functions = get_functions(
        # quadratic_function(nums=1000, x_range=[-5, 5], b=300),
        # quadratic_function(a=-1, nums=1000, x_range=[-5, 5], b=-120),
        # quadratic_function(a=2, nums=1000, x_range=[25, 37], b=10, c=-20),
        # quadratic_function(nums=1000, x_range=[12, 17], c=-12, b=-10),
        # quadratic_function(nums=1000, x_range=[0, 20], c=-10, b=-350),
        gaussian(nums=1000, y_mu=-25, x_mu=-10),
        gaussian(nums=1000, y_mu=-100, x_mu=20),
        gaussian(nums=1000, y_mu=100, y_sigma=10, x_mu=-5),
        gaussian(nums=1000, y_mu=500, x_sigma=1, y_sigma=6, x_mu=50),
        gaussian(nums=1000, y_mu=500, x_sigma=1, y_sigma=15, x_mu=-10)
    )

    dataset = Function_Generator_Dataset(functions=functions, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Basic_Model()
    model.cuda()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.0e-4)

    for epoch in range(args.epoch):
        for step, (b_x, b_y) in enumerate(dataloader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()

            logits = model.forward(b_x)
            loss = model.cal_distance_loss(logits, b_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print('train loss: {}'.format(str(loss.item())))

if __name__ == '__main__':
    args = parse_args()
    train()
