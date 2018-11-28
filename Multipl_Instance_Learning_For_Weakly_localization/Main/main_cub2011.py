import sys

sys.path.append('../')

import argparse
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dataloader.Bi_Custom_Dataset import *

plt.switch_backend('agg')

import pickle
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=20)

    parser.add_argument('--data_root_path', type=str, default='../../../../../Data/CUB2011/CUB_200_2011/')
    parser.add_argument('--bi_train_data_path', type=str, default='../Save/CUB2011/bi_train_data.csv')
    parser.add_argument('--bi_val_data_path', type=str, default='../Save/CUB2011/bi_val_data.csv')
    parser.add_argument('--bi_test_data_path', type=str, default='../Save/CUB2011/bi_test_data.csv')

    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    return args


# def test():
#     dataset = CUB(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
#                   test_data=args.test_data_path, mode='test')
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
#
#     model = torch.load('../Save/CUB2011/model/model_vgg_backbone.pt')
#     model.cuda()
#
#     class_results = []
#     class_labels = []
#
#     bbox_results = []
#     bbox_labels = []
#     with torch.no_grad():
#         for step, (b_name, b_x, b_y) in enumerate(dataloader):
#             b_x = b_x.cuda()
#             b_y = b_y.cuda()
#
#             cam, logits = model.forward(b_x)
#
#             final_logit = logits[0] + logits[-1]
#
#             class_results.extend(np.argmax(final_logit.data.cpu().numpy(), axis=-1))
#             class_labels.extend(b_y.data.cpu().numpy())
#
#     print(
#         'test class acc: ' + str(np.sum(np.array(class_results) == np.array(class_labels)) / float(len(class_labels))))


def bi_train():
    dataset = CUB_BI(root_dir=args.data_root_path, bi_bird_data='../Save/bi_bird_data.csv',
                     bi_nature_data='../Save/bi_nature_data.csv', mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = CUB_BI(root_dir=args.data_root_path, bi_bird_data='../Save/bi_bird_data.csv',
                         bi_nature_data='../Save/bi_nature_data.csv', mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size * 10, shuffle=False)

    # model = get_vgg(pretrained=True, num_classes=340, is_RGB=False)
    # model.cuda()

    for epoch in range(args.epoch):
        # cur_opt = decrease_lr_by_epoch(epoch, model, args)
        for step, (b_name, b_x, b_y) in enumerate(dataloader):
            print()

if __name__ == '__main__':
    args = parse_args()
    bi_train()
