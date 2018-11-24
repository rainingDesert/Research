import torch
import numpy as np
import pandas as pd

'''
单分类问题acc
'''


def cal_single_label_acc(result, label, top_k=1):
    label = label.cpu().data.numpy()

    if top_k > 1:
        max_index = np.argsort(result.cpu().data.numpy(), axis=-1)[-1 * top_k:]

        return np.sum(max_index == label) / float(len(label))
    else:
        max_index = np.argmax(result.cpu().data.numpy(), axis=-1)

        return np.sum(max_index == label) / float(len(label))


def check_x_to_y(img_name, b_y, args):
    train_csv = pd.read_csv(args.train_data_path)
    img_path = list(train_csv[train_csv['img_name'].isin(img_name)]['path'])
    true_name = np.array([int(item.split('.')[0]) for item in img_path])
    return (true_name == b_y.data.numpy()).all()
