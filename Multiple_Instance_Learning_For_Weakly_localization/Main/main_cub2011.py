import sys

sys.path.append('../')

import matplotlib

# matplotlib.use('agg')
import argparse
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

from dataloader.Bi_self_dataset import CUB_BI
from dataloader.classify_Custom_dataset import *

from model.linear_vgg import get_linear_vgg
from model.classify_vgg import get_classify_vgg
from model.bi_vgg import get_bi_vgg
from model.semi_vgg import get_semi_vgg

from Main.custom_optim import *

from functions.metrics import cal_single_label_acc, cal_sigmoid_acc, plot_raw_cam_bi, get_raw_imgs, cal_iou, \
    get_max_binary_area, get_iou

# plt.switch_backend('agg')

import pickle
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--data_root_path', type=str, default='../../../../../Data/CUB2011/CUB_200_2011/')
    parser.add_argument('--bi_train_data_path', type=str, default='../Save/CUB2011/bi_train_data.csv')
    parser.add_argument('--bi_val_data_path', type=str, default='../Save/CUB2011/bi_val_data.csv')
    parser.add_argument('--bi_test_data_path', type=str, default='../Save/CUB2011/bi_test_data.csv')

    parser.add_argument('--train_data_path', type=str, default='../Save/train_data.csv')
    parser.add_argument('--val_data_path', type=str, default='../Save/val_data.csv')
    parser.add_argument('--test_data_path', type=str, default='../Save/test_data.csv')

    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    return args


def bi_self_train():
    dataset = CUB_BI(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
                     test_data=args.test_data_path, nature_data=args.data_root_path + 'natures/', mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = CUB_BI(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
                         test_data=args.test_data_path, nature_data=args.data_root_path + 'natures/', mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    loss_func = torch.nn.BCELoss()

    model = get_bi_vgg(pretrained=True, freeze_vgg=True)
    model.cuda()

    best_test_acc = 0.0
    best_iou = 0.0
    for epoch in range(args.epoch):
        cur_opt = decrease_lr_by_epoch(epoch, model, args)
        for step, (b_name, b_x, b_y) in enumerate(dataloader):
            b_x = b_x.cuda()
            b_y = b_y.type(torch.FloatTensor).cuda()

            logits, y_hat = model.forward(b_x)
            loss = loss_func(logits, b_y)

            cur_opt.zero_grad()
            loss.backward()
            cur_opt.step()

            train_acc = cal_sigmoid_acc(y_hat, b_y)

            print('mode: train | ' + 'epoch ' + str(epoch) + ' | ' + 'loss: ' + str(
                loss.cpu().data.numpy()) + 'train acc: ' + str(train_acc) + ' | best_test_acc: ' + str(
                best_test_acc) + ' | best iou: ' + str(best_iou))

            if step % 50 == 0:
                model.to_inference()
                acc = 0.0
                iou = 0.0
                for step, (b_name, b_x, b_y, bbox) in enumerate(val_dataloader):
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()

                    logits, y_hat, x_grad = model.forward(b_x)
                    acc = cal_sigmoid_acc(y_hat, b_y)
                    norm_cam_np = get_max_binary_area(
                        model.norm_grad_2_binary(x_grad.detach()).squeeze().data.numpy())

                    for i in range(norm_cam_np.shape[0]):
                        iou += get_iou(norm_cam_np[i], bbox[i])

                if acc > best_test_acc:
                    best_test_acc = acc

                if (iou / b_x.size(0)) > best_iou:
                    best_iou = (iou / b_x.size(0))

                torch.save(model, '../Save/model/bi_cam_background.pt')
                model.close_inference()


def bi_self_test():
    dataset = CUB_BI(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
                     test_data=args.test_data_path, nature_data=args.data_root_path + 'natures/', mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    bi_model = get_bi_vgg(pretrained=True, trained=True, inference=True)
    bi_model.cuda()

    classify_result = []
    iou_result = []
    print('start test')
    for step, (b_name, b_x, b_y) in enumerate(tqdm(dataloader)):
        b_x = b_x.cuda()
        b_y = b_y.cuda()

        bi_logits, y_hat, bi_x_grad = bi_model.forward(b_x)
        img_path, raw_img, bbox = get_raw_imgs(b_name, dataset)
        final_cam = F.upsample(bi_x_grad, size=[raw_img.size[1], raw_img.size[0]], mode='bilinear', align_corners=True)

        norm_cam_np = get_max_binary_area(bi_model.norm_grad_2_binary(final_cam.detach()).squeeze().data.numpy())
        draw_raw, iou = cal_iou(norm_cam_np, bbox, raw_img.copy())
        # plot_raw_cam_bi(raw_img, draw_raw, bi_x_grad, norm_cam_np)

        iou_result.append(iou > 0.5)
        classify_result.append(y_hat.item() == b_y.item())

    print('cls acc: ' + str(np.array(classify_result).mean()))
    print('iou: ' + str((np.array(iou_result) * np.array(classify_result)).mean()))


def classify_train():
    dataset = CUB_Cls(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
                      test_data=args.test_data_path, mode='train', img_size=224)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = CUB_Cls(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
                          test_data=args.test_data_path, mode='val', img_size=224)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size * 10, shuffle=False)

    bi_model = get_bi_vgg(pretrained=True, trained=True, inference=True)
    bi_model.to_inference()
    bi_model.cuda()

    model = get_semi_vgg(pretrained=True, trained=False, inference=False)
    model.cuda()

    cls_loss = torch.nn.CrossEntropyLoss()

    best_test_acc = 0.0
    for epoch in range(args.epoch):
        cur_opt = decrease_lr_by_epoch(epoch, model, args, fine_tune=True)
        for step, (b_name, b_x, b_y) in enumerate(dataloader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()

            logits, cam = model.forward(b_x)
            bi_logits, y_hat, bi_x_grad = bi_model.forward(b_x)
            norm_largest_grad = torch.from_numpy(
                get_max_binary_area(bi_model.norm_grad_2_binary(bi_x_grad).data.numpy())).type(torch.FloatTensor).cuda()

            classify_loss = cls_loss(logits, b_y)
            semi_loss = model.cal_element_loss(norm_largest_grad, cam)

            train_acc = cal_single_label_acc(logits, b_y)
            cur_opt.zero_grad()
            classify_loss.backward(retain_graph=True)
            model.freeze_vgg = True
            semi_loss.backward()
            cur_opt.step()
            model.freeze_vgg = False

            # if step % 500 == 0:
            #     plt.imshow(b_x[0].cpu().data.numpy().transpose(1, 2, 0))  # raw img
            #     plt.savefig('../Save/images/raw.png')
            #     plt.imshow(bi_x_grad[0].squeeze().cpu().data.numpy(), cmap='hot')  # bi_x_grad
            #     plt.savefig('../Save/images/bi_x_grad.png')
            #     plt.imshow(norm_largest_grad[0].cpu().data.numpy(), cmap='hot')  # norm_largest_grad
            #     plt.savefig('../Save/images/norm_largest_grad.png')
            #     plt.imshow(cam[0].squeeze().cpu().data.numpy(), cmap='hot')  # cam
            #     plt.savefig('../Save/images/cam.png')

            print('mode: train | ' + 'epoch ' + str(epoch) + ' | ' + 'cls loss: ' + str(
                classify_loss.item()) + ' bce loss: ' + str(semi_loss.item()) + ' train acc: ' + str(
                train_acc) + ' | best_test_acc: ' + str(best_test_acc))

            if step % 50 == 0:
                with torch.no_grad():
                    acc = 0.0
                    for step, (b_name, b_x, b_y) in enumerate(val_dataloader):
                        b_x = b_x.cuda()
                        b_y = b_y.cuda()

                        logits, cam = model.forward(b_x)
                        acc = cal_single_label_acc(logits, b_y)

                    if acc >= best_test_acc:
                        best_test_acc = acc
                        torch.save(model, '../Save/model/semi_model_vgg_back.pt')


def classify_test():
    dataset = CUB_Cls(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
                      test_data=args.test_data_path, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    bi_model = get_bi_vgg(pretrained=True, trained=True, inference=True)
    bi_model.to_inference()
    bi_model.cuda()

    semi_model = get_semi_vgg(pretrained=True, trained=True, inference=False)
    semi_model.cuda()

    # multi_branch_model=get_multi_branch_vgg(pretrained=True,trained=True)
    # multi_branch_model.cuda()

    classify_results = []
    local_results = []
    for step, (b_name, b_x, b_y) in enumerate(tqdm(dataloader)):
        b_x = b_x.cuda()
        b_y = b_y.cuda()

        logits, cam = semi_model.forward(b_x)
        bi_logits, y_hat, bi_x_grad = bi_model.forward(b_x)

        img_path, raw_img, bbox = get_raw_imgs(b_name, dataset)
        final_cam = F.upsample(cam, size=[raw_img.size[1], raw_img.size[0]], mode='bilinear', align_corners=True)
        norm_cam_np = get_max_binary_area(semi_model.norm_grad_2_binary(final_cam.detach()).squeeze().data.numpy())

        draw_raw, iou = cal_iou(norm_cam_np, bbox, raw_img.copy())

        # plot_raw_cam_bi(raw_img, draw_raw, final_cam, norm_cam_np)
        classify_results.append(torch.argmax(logits, dim=-1).item() == b_y.item())
        local_results.append(iou > 0.5)

    print('cls acc: ' + str(np.array(classify_results).mean()))
    print('iou: ' + str((np.array(local_results) * np.array(classify_results)).mean()))


# exper
def get_obj_with_no_train():
    dataset = CUB_Cls(root_dir=args.data_root_path, train_data=args.train_data_path, val_data=args.val_data_path,
                      test_data=args.test_data_path, mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    bi_model = get_bi_vgg(pretrained=True, trained=True, inference=True)
    bi_model.to_inference()
    bi_model.cuda()

    linear_model = get_linear_vgg(pretrained=True, inference=True)
    linear_model.cuda()

    for step, (b_name, b_x, b_y) in enumerate(dataloader):
        b_x = b_x.cuda()
        b_y = b_y.cuda()

        F_b_x = F.upsample(b_x, size=(224, 224), mode='bilinear', align_corners=True)

        img_path, raw_img, bbox = get_raw_imgs(b_name, dataset)
        bi_logits, y_hat, bi_x_grad = bi_model.forward(b_x)
        bi_norm_grad = get_max_binary_area(bi_model.norm_grad_2_binary(bi_x_grad).numpy())
        norm_x_grad = linear_model.forward(F_b_x)

        plt.imshow(raw_img)
        plt.show()
        plt.imshow(bi_norm_grad[0])
        plt.show()
        plt.imshow(norm_x_grad[0].cpu().data.numpy())
        plt.show()
        print()


if __name__ == '__main__':
    # args = parse_args()
    # bi_self_test()
    # bi_self_train()
    # bi_self_test()
    # classify_train()
    # multi_branch_classify_train()
    # get_binary_cam_no_train()
    # classify_test()
    # bi_self_train()
    # bi_self_test()
    # classify_train()
    # get_obj_with_no_train()
    # classify_test()

    def swap_rows(m):
        new_m = np.vstack((m, m[0]))
        new_m = np.delete(new_m, (0), 0)

        return new_m


    matrix1 = torch.from_numpy(np.identity(3)).type(torch.FloatTensor)
    matrix2 = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [2, 0, 1]])).type(torch.FloatTensor)


    def cal_distance(m1, m2):
        trans = []
        for i in range(len(m1[0])):
            if len(trans) == 0:
                trans.append(np.identity(3))
            else:
                trans.append(swap_rows(trans[-1].copy()))

        result = torch.mm(torch.from_numpy(np.array(trans).reshape(len(trans) * len(trans[1]), len(trans[2]))).type(
            torch.FloatTensor), m2)
        matrix1_3d = m1.unsqueeze(0).repeat(3, 1, 1).view(-1, 3)
        tmp = torch.sum((matrix1_3d - result) ** 2, dim=-1) ** 0.5

        return tmp.view(3, 3).transpose(1, 0)


    tmp = cal_distance(matrix1, matrix2)
    tmp2 = cal_distance(matrix2, matrix1)
    print(tmp)
    print(tmp2)
