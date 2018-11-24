import sys

from torch.nn import MultiLabelSoftMarginLoss
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dataloader.CUB2011.Custom_Dataset import CUB
from dataloader.Doodle.Custom_Dataset import Doodle
from model.Doodle_model.resnet_50 import *
from model.Doodle_model.squeezenet1_1 import _squeezenet
import argparse
from Main.custom_optim import *
from functions.metrics import *
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--data_root_path', type=str, default='../../../../../../Data/Doodle/')

    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    return args


'''
正常使用resnet-pretrained进行训练
获得分类准确度 CAM
使用CUB数据集
'''


def normal_cub():
    dataset = Doodle(mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = Doodle(mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


def doodle_model():
    dataset = Doodle(train_csv_path='../Save/Doodle/train_list.csv', val_csv_path='../Save/Doodle/val_list.csv',
                     test_csv_path='../Save/Doodle/test_list.csv', args=args, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = Doodle(train_csv_path='../Save/Doodle/train_list.csv', val_csv_path='../Save/Doodle/val_list.csv',
                         test_csv_path='../Save/Doodle/test_list.csv', args=args, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size * 10, shuffle=False)

    model = _squeezenet(num_classes=340)
    model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = get_regular_optimizer(args, model)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    test_acc = 0.0
    for epoch in range(args.epoch):
        for step, (b_name, b_x, b_y) in enumerate(dataloader):
            if not check_x_to_y(b_name,b_y):
                continue

            b_x = b_x.cuda()
            b_y = b_y.cuda()

            result = model.forward(b_x)
            loss = loss_func(result, b_y)
            acc = cal_single_label_acc(result, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.cpu().data.numpy())
            train_acc.append(acc)

            print('mode: train | ' + 'epoch ' + str(epoch) + ' | ' + 'loss: ' + str(
                loss.cpu().data.numpy()) + ' | acc: ' + str(acc))

            if step % 50 == 0:
                with torch.no_grad():
                    for step, (b_name, b_x) in enumerate(val_dataloader):
                        b_x = b_x.cuda()
                        b_y = b_y.cuda()

                        result = model.forward(b_x)
                        loss = loss_func(result, b_y)
                        acc = cal_single_label_acc(result, b_y)

                        print(
                            'mode: val | ' + 'epoch ' + str(epoch) + ' | ' + 'loss: ' + str(
                                loss.cpu().data.numpy()) + ' | acc:' + str(
                                acc))

                        if acc > test_acc:
                            test_acc = acc
                            # torch.save(model, '../Save/Doodle/model/model.pt')


def doodle_test():
    dataset = Doodle(train_csv_path='../Save/Doodle/train_list.csv', val_csv_path='../Save/Doodle/val_list.csv',
                     test_csv_path='../Save/Doodle/test_list.csv', args=args, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = torch.load('../Save/Doodle/model/model.pt')
    model.cuda()

    word_list = np.array(pd.read_csv(args.data_root_path + 'train/word_list.csv')['word'])

    result_arr = []
    for step, (b_name, b_x) in enumerate(dataloader):
        b_x = b_x.cuda()

        result = model.forward(b_x)
        np_result = result.cpu().data.numpy()
        result_arr.append([b_name[0], ' '.join(list(word_list[np.argsort(np_result)[-3:]]))])
        # result_arr.append([str(b_name[0].cpu().data.numpy()),])
        # result_map[str(b_name[0].cpu().data.numpy())] = np.argsort(np_result)[-3:]

    pd.DataFrame(data=result_arr, columns=['key_id', 'word']).to_csv('../Save/Doodle/result/result.csv', index=False)


if __name__ == '__main__':
    args = parse_args()
    doodle_test()
