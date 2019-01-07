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

    opt = optim.SGD([{'params': weight_list, 'lr': lr / 10},
                     {'params': bias_list, 'lr': lr / 5},
                     {'params': last_weight_list, 'lr': lr},
                     {'params': last_bias_list, 'lr': lr * 2}], momentum=0.9, weight_decay=0.0001, nesterov=False)

    return opt
