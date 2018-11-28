import sys

sys.path.append('../')

import argparse
from torch.utils.data import Dataset, DataLoader
from dataloader.CUB2011.Custom_Dataset import *
from model.Zhang_CVPR_2018.resnet50 import *
from Main.custom_optim import *
from functions.metrics import *
import matplotlib.pyplot as plt
from model.Zhang_CVPR_2018.vgg import *

plt.switch_backend('agg')

import pickle
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=20)

    parser.add_argument('--data_root_path', type=str, default='../../../../../../Data/CUB2011/CUB_200_2011/')
    parser.add_argument('--train_data_path', type=str, default='../Save/CUB2011/train_data.csv')
    parser.add_argument('--val_data_path', type=str, default='../Save/CUB2011/val_data.csv')
    parser.add_argument('--test_data_path', type=str, default='../Save/CUB2011/test_data.csv')

    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    return args


def test():
    dataset = CUB(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
                  test_data=args.test_data_path, mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = torch.load('../Save/CUB2011/model/model_vgg_backbone.pt')
    model.cuda()

    class_results = []
    class_labels = []

    bbox_results = []
    bbox_labels = []
    with torch.no_grad():
        for step, (b_name, b_x, b_y) in enumerate(dataloader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()

            cam, logits = model.forward(b_x)

            final_logit = logits[0] + logits[-1]

            class_results.extend(np.argmax(final_logit.data.cpu().numpy(), axis=-1))
            class_labels.extend(b_y.data.cpu().numpy())

    print(
        'test class acc: ' + str(np.sum(np.array(class_results) == np.array(class_labels)) / float(len(class_labels))))


def train():
    dataset = CUB(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
                  test_data=args.test_data_path, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = CUB(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
                      test_data=args.test_data_path, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size * 10, shuffle=False)

    if os.path.exists('../Save/CUB2011/model/model_vgg_backbone.pt'):
        model = torch.load('../Save/CUB2011/model/model_vgg_backbone.pt')
    else:
        model = get_vgg(pretrained=True, num_classes=200, threshold=0.6)
    model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    best_test_acc = 0.0
    for epoch in range(args.epoch):
        cur_opt = decrease_lr_by_epoch(epoch, model, args)
        for step, (b_name, b_x, b_y) in enumerate(dataloader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()

            cams, logits = model.forward(b_x, label=b_y)
            loss = loss_func(logits[-1], b_y) + loss_func(logits[0], b_y)

            acc = cal_single_label_acc(logits[0], b_y.detach())

            cur_opt.zero_grad()
            loss.backward()
            cur_opt.step()

            train_loss.append(loss.detach().cpu().data.numpy())
            train_acc.append(acc)

            print('mode: train | ' + 'epoch ' + str(epoch) + ' | ' + 'loss: ' + str(
                loss.cpu().data.numpy()) + ' | acc: ' + str(acc))

            if step % 50 == 0:
                with torch.no_grad():
                    for step, (b_name, b_x, b_y) in enumerate(val_dataloader):
                        b_x = b_x.cuda()
                        b_y = b_y.cuda()

                        cams, logits = model.forward(b_x, label=b_y)
                        loss = loss_func(logits[-1], b_y) + loss_func(logits[0], b_y)
                        acc = cal_single_label_acc(logits[0], b_y)

                        print(
                            'mode: val | ' + 'epoch ' + str(epoch) + ' | ' + 'loss: ' + str(
                                loss.cpu().data.numpy()) + ' | acc:' + str(
                                acc))

                        test_loss.append(loss.cpu().data.numpy())
                        test_acc.append(acc)

                        if acc > best_test_acc:
                            best_test_acc = acc
                            torch.save(model, '../Save/CUB2011/model/model_vgg_backbone.pt')

                        obs_img_name = b_name[0]

                        val_csv = pd.read_csv(args.val_data_path)
                        img_path = list(val_csv[val_csv['img_name'] == obs_img_name]['path'])[0]
                        raw_img = Image.open(args.data_root_path + 'images/' + img_path).convert('RGB')
                        raw_img.save('../Save/CUB2011/jpg/obs_raw_img.jpg')

                        for index, target_cam in enumerate(cams):
                            cam = target_cam.cpu().data.numpy()[0][np.argmax(logits[index].cpu().data.numpy()[0])]

                            plt.imshow(cam)
                            plt.savefig('../Save/CUB2011/jpg/obs_cam' + '_' + str(index) + '.jpg')

                        # for i in range(cam.cpu().data.numpy()[0].shape[0]):
                        #     plt.imshow(cam.cpu().data.numpy()[0][np.argmax(logits.cpu().data.numpy()[0][i])])
                        #     plt.savefig('../Save/CUB2011/jpg/all_cam/' + str(i) + '.jpg')

        with open('../Save/CUB2011/train.pkl', 'wb') as f:
            pickle.dump([train_loss, train_acc, test_loss, test_acc], f)


if __name__ == '__main__':
    args = parse_args()
    test()

    # import torch
    # from torch.autograd import Variable
    #
    # x = torch.Tensor([[1., 2., 3.], [4., 5., 6.]])
    # # x=Variable(x,requires_grad=True)
    # x.requires_grad = True
    # y = x + 2
    # z = y * y * 3
    # out = z.mean()
    #
    # out.backward(None)
    #
    # print(out)
