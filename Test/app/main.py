from torch.utils.data import DataLoader
from dataloader.loader import Binary_dataset
import argparse

from models.binary_vgg_cls import get_binary_vgg
from app.optim import *
from app.functions import *

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--data_root_path', type=str, default='/media/kzy/tt/Data/imgnet_localization/')
    parser.add_argument('--csv_path', type=str, default='../save/data.csv')

    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--train_log', type=str, default='../save/train/train_log.txt')

    args = parser.parse_args()
    return args


'''
使用二分类观察同一object与不同object进行分类的CAM
'''


def binary_train(args):
    dataset = Binary_dataset(args=args, query_cls='n02017213')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = get_binary_vgg(pretrained=True)
    loss_func = torch.nn.CrossEntropyLoss()

    with open(args.train_log, 'w') as f:
        f.truncate()

    for epoch in range(args.epoch):
        opt = get_finetune_optimizer(args, model)

        train_result = []
        train_label = []
        val_result = []
        val_label = []
        for step, (img_id, img, label, bbox) in enumerate(dataloader):
            img = img.cuda()
            label = label.cuda()

            logits, cam = model.forward(img)
            loss = loss_func(logits, label)
            acc = cal_acc(logits, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print('epoch:{} train loss:{} train acc:{}'.format(epoch, loss, acc))

            train_result.extend(torch.argmax(logits, dim=-1).cpu().data.numpy())
            train_label.extend(label.cpu().data.numpy())

        train_acc = np.mean(np.array(train_result) == np.array(train_label))

        # validation
        dataset.to_val()
        val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for step, (img_id, img, label, bbox) in enumerate(val_dataloader):
            img = img.cuda()
            label = label.cuda()

            logits, cam = model.forward(img)
            val_result.extend(torch.argmax(logits, dim=-1).cpu().data.numpy())
            val_label.extend(label.cpu().data.numpy())

            if step == 0:
                save_imgs_cams_by_id(args, img_id, logits,cam, dataset)

        val_acc = np.mean(np.array(val_result) == np.array(val_label))

        with open(args.train_log, 'a') as f:
            f.writelines('{}-{}'.format(train_acc, val_acc) + '\n')

        dataset.to_train()


if __name__ == '__main__':
    args = parse_args()
    binary_train(args=args)
