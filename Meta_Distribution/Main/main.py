import sys

sys.path.append('../')

from Dataset.Function_Generator_Dataset import Function_Generator_Dataset, get_functions, quadratic_function, gaussian
import argparse
from torch.utils.data import DataLoader
from models.basic_model import Basic_Model
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    return args


def train():
    q_c = [0.6, 1.2, 1.8, 2.4, 3.0]
    q_sigma = 0.25

    functions_train = get_functions(
        # quadratic_function(nums=1000, x_range=[q_c[0]-q_sigma, q_c[0]+q_sigma], c=q_c[0], b=2.4,a=10),
        # quadratic_function(nums=1000, x_range=[q_c[1]-q_sigma, q_c[1]+q_sigma], c=q_c[1], b=1.2,a=10),
        # quadratic_function(nums=1000, x_range=[q_c[2]-q_sigma, q_c[2]+q_sigma], c=q_c[2], b=3.2,a=10),
        # quadratic_function(nums=1000, x_range=[q_c[3]-q_sigma, q_c[3]+q_sigma], c=q_c[3], b=0.6,a=10),
        # quadratic_function(nums=1000, x_range=[q_c[4]-q_sigma, q_c[4]+q_sigma], c=q_c[4], b=1.8,a=10),

        # gaussian(nums=1000, y_mu=0.8, x_mu=1, x_sigma=0.1, y_sigma=0.01),
        # gaussian(nums=1000, y_mu=1.5, x_mu=0.5, x_sigma=0.1, y_sigma=0.01),
        # gaussian(nums=1000, y_mu=2.3, x_mu=0.2, x_sigma=0.1, y_sigma=0.01),
        # gaussian(nums=1000, y_mu=1.9, x_mu=2.6, x_sigma=0.1, y_sigma=0.01),
        gaussian(nums=1000, y_mu=0, x_mu=0, x_sigma=0.25, y_sigma=0.025),
        gaussian(nums=1000, y_mu=0.2, x_mu=0, x_sigma=0.25, y_sigma=0.025),
        gaussian(nums=1000, y_mu=0.4, x_mu=0, x_sigma=0.25, y_sigma=0.025),

        gaussian(nums=1000, y_mu=-0.6, x_mu=0, x_sigma=0.25, y_sigma=0.025),
        # gaussian(nums=1000, y_mu=0.3, x_mu=0, x_sigma=0.1, y_sigma=0.05),
        # gaussian(nums=1000, y_mu=1, x_mu=2, x_sigma=0.01, y_sigma=0.1),
        gaussian(nums=1000, y_mu=0, x_mu=0, x_sigma=0.025, y_sigma=0.25),
        gaussian(nums=1000, y_mu=0, x_mu=-0.2, x_sigma=0.025, y_sigma=0.25),
        gaussian(nums=1000, y_mu=0, x_mu=-0.4, x_sigma=0.025, y_sigma=0.25),
        gaussian(nums=1000, y_mu=0, x_mu=0.2, x_sigma=0.025, y_sigma=0.25),
        gaussian(nums=1000, y_mu=0, x_mu=0.4, x_sigma=0.025, y_sigma=0.25),

        # gaussian(nums=1000, y_mu=2.3, x_mu=1.2, x_sigma=0.01, y_sigma=0.1),
        # gaussian(nums=1000, y_mu=1.9, x_mu=0.6, x_sigma=0.01, y_sigma=0.1),
        # gaussian(nums=1000, y_mu=2.5, x_mu=0.3, x_sigma=0.01, y_sigma=0.1)
    )

    dataset_train = Function_Generator_Dataset(x_nums=100, y_nums=100, iter_nums=1000)
    # dataset_train.plot_all_points()

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

        # quadratic_function(nums=1000, x_range=[-60, -80], c=70, b=20),
        # gaussian(nums=1000, y_mu=0.8, x_mu=1, x_sigma=0.1, y_sigma=0.1),
        # gaussian(nums=1000, y_mu=1.2, x_mu=1.0, x_sigma=0.1, y_sigma=0.01)
        gaussian(nums=1000, y_mu=-0.2, x_mu=0, x_sigma=0.25, y_sigma=0.025),
    )

    infer_sample = 20

    dataset_test = Function_Generator_Dataset(x_nums=100, y_nums=100, iter_nums=1000)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    model = Basic_Model(input_dims=infer_sample * 2)
    model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        for step, (b_x, b_y) in enumerate(dataloader_train):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            random_index = torch.randint(0, 100, (infer_sample,)).long()
            b_x = b_x[:, random_index, :]
            # b_x = b_x.view(b_x.size(0), -1)

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
                    # print(b_x.size())

                    random_index = torch.randint(0, 100, (infer_sample,)).long()
                    b_x = b_x[:, random_index, :]
                    logits = model(b_x)

                    logits = torch.unsqueeze(logits[0], dim=0)
                    b_x = torch.unsqueeze(b_x[0], dim=0)
                    b_y = torch.unsqueeze(b_y[0], dim=0)
                    # print(b_y.size())
                    model.plot_reuslts_points_and_gt(logits.cpu().data.numpy(), b_x.cpu().data.numpy(),
                                                     b_y.cpu().data.numpy())
                    print('fake_std_x: {}'.format(float(torch.std(logits[0, :, 0]))))
                    print('real_std_x: {}'.format(float(torch.std(b_y[0, :, 0]))))
                    print('fake_std_y: {}'.format(float(torch.std(logits[0, :, 1]))))
                    print('real_std_y: {}'.format(float(torch.std(b_y[0, :, 1]))))
                    # print(logits)

                    break


if __name__ == '__main__':
    args = parse_args()
    train()
