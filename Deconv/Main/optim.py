import torch.optim as optim


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
                     {'params': last_bias_list, 'lr': lr * 20}], momentum=0.9, weight_decay=0.0005,
                    nesterov=True)

    return opt


def get_deconv_finetune_optimizer(args, model):
    lr = args.lr

    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list = []

    decov_weight_list = []
    deconv_bias_list = []

    for name, value in model.named_parameters():
        if 'features' in name:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)
        elif 'deconv' in name:
            if 'weight' in name:
                decov_weight_list.append(value)
            elif 'bias' in name:
                deconv_bias_list.append(value)
        else:
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)

    opt = optim.SGD([{'params': weight_list, 'lr': lr / 10},
                     {'params': bias_list, 'lr': lr / 5},
                     {'params': last_weight_list, 'lr': lr},
                     {'params': last_bias_list, 'lr': lr * 2},
                     {'params': decov_weight_list, 'lr': lr * 5},
                     {'params': deconv_bias_list, 'lr': lr * 10}], momentum=0.9, weight_decay=0.0001,
                    nesterov=False)

    return opt
