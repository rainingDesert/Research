import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch
import math


def get_finetune_optimizer(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list = []
    for name, value in model.named_parameters():
        if 'cls' in name:
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    opt = optim.SGD([{'params': weight_list, 'lr': lr},
                     {'params': bias_list, 'lr': lr * 2},
                     {'params': last_weight_list, 'lr': lr * 10},
                     {'params': last_bias_list, 'lr': lr * 20}], momentum=0.9, weight_decay=0.0001, nesterov=True)

    return opt


def decrease_lr_by_epoch(epoch, model, args):
    cur_lr = args.lr * math.pow(0.9, epoch)

    opt = optim.Adam(model.parameters(), lr=cur_lr, betas=(0.9, 0.999), weight_decay=10e-5)

    return opt


def get_regular_optimizer(args, model):
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.0e-4)
