import sys

sys.path.append('../')

from torch.utils.data import DataLoader
import argparse
import pandas as pd
import pickle
import torchvision
import torch.utils.data as Data
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from Model.base_vgg import get_base_vgg_model
from Model.deconv_vgg2 import get_vgg_deconv2_model
from Model.vgg_auto_encoder import get_vgg_auto_encoder_model
from Model.dilation_vgg import get_dilation_vgg_model

from Dataloader.loader import CUB_Loader
from Function.helper import *
from Function.cus_plot import *
from Main.optim import *

from graph_seg.main import segment

import gc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--data_root_path', type=str, default='/Users/guofengcui/study/UofR/research/Ideas/CUB/'),
    parser.add_argument('--csv_path', type=str, default='../Save/data.csv')
    parser.add_argument('--result_path', type = str, default = '../Save/{}_result.pkl'.format('bas_vgg'))
    parser.add_argument('--check_path', type=str, default='../Save/{}_check.pkl'.format('bas_vgg'))
    parser.add_argument('--train_img', type=str,
                        default='../Save/imgs/{}_train_process.png'.format('bas_vgg')),
    parser.add_argument('--save_model_path', type=str, default='../Save/models/{}_model.pt'.format('bas_vgg')),
    parser.add_argument('--test_cam_path', type=str, default='../Save/imgs/{}_test_cam.png'.format('bas_vgg')),

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_img_size', type=int, default=224)
    parser.add_argument('--class_nums', type=int, default=2)
    parser.add_argument('--continue_train', type=bool, default=False)

    parser.add_argument('--gpu', type = bool, default = False)
    parser.add_argument('--cls', type = int, default = 0)
    parser.add_argument('--tot_cls', type = int ,default = 200)

    args = parser.parse_args()
    return args


def base_vgg_cls():
    # 初始化
    if args.gpu:
        torch.cuda.empty_cache()

    epoch = 0
    train_acc_arr = []
    val_acc_arr = []

    # 加载数据
    dataset = CUB_Loader(args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 加载模型
    model = get_base_vgg_model(args=args)

    # 加载参数
    loss_func = torch.nn.CrossEntropyLoss()

    # 训练
    if args.continue_train:
        model.load_state_dict(torch.load(args.train_model))
        check_dict = load_check_point(args=args)
        epoch = check_dict['epoch']
        train_acc_arr = check_dict['train_acc_arr']
        val_acc_arr = check_dict['val_acc_arr']

    while epoch < args.epoch:
        opt = get_finetune_optimizer(args, model)

        train_result = []
        train_label = []
        val_result = []
        val_label = []

        for step, (img_id, img, label, bbox) in enumerate(dataloader):
            if args.gpu:
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

        train_acc_arr.append(np.mean(np.array(train_result) == np.array(train_label)))

        # validation
        # dataset.to_val()
        # val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        # for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
        #     if args.gpu:
        #         img = img.cuda()
        #         label = label.cuda()

        #     logits, cam = model.forward(img)
        #     val_result.extend(torch.argmax(logits, dim=-1).cpu().data.numpy())
        #     val_label.extend(label.cpu().data.numpy())

            # if step == 0:
            #     target_cls = torch.argmax(logits, dim=-1)

                # plot_dict = dict()
                # plot_dict['raw_imgs'] = get_raw_imgs_by_id(args, img_id[:5], dataset)
                # target_cams = []
                # for i in range(5):
                #     raw_img_size = plot_dict['raw_imgs'][i].size
                #     target_cam = cam[i][target_cls[i]].unsqueeze(0).unsqueeze(0).detach().cpu().data
                #     up_target_cam = F.upsample(target_cam, size=(raw_img_size[1], raw_img_size[0]), mode='bilinear',
                #                                align_corners=True)
                #     target_cams.append(up_target_cam.squeeze())

                # plot_dict['cams'] = target_cams
                # plot_different_figs(args, plot_dict)

        # val_acc_arr.append(np.mean(np.array(val_result) == np.array(val_label)))

        # if len(val_acc_arr) == 1 or val_acc_arr[-1] >= val_acc_arr[-2]:
        #     torch.save(model.state_dict(), args.save_model_path)

        # plot
        # plot_train_process(args, [train_acc_arr, val_acc_arr])

        # save check point
        epoch += 1
        # save_check_point(args=args, check_dict={
        #     'epoch': epoch,
        #     'train_acc_arr': train_acc_arr,
        #     'val_acc_arr': val_acc_arr
        # })

        dataset.to_train()

    return model, dataset

def base_vgg_bbx(model, dataset):
    val_result = []
    val_label = []
    val_ious = []
    val_acc_arr = []
    val_acc_iou = []

    # load validation
    dataset.to_val()
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # do validation
    for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
        if args.gpu:
            img = img.cuda()
            label = label.cuda()

        logits, cam = model.forward(img)
        val_result.extend(torch.argmax(logits, dim=-1).cpu().data.numpy())
        val_label.extend(label.cpu().data.numpy())

        # get bounding box
        imgs = get_raw_imgs_by_id(args, img_id, dataset)
        target_cls = torch.argmax(logits, dim=-1)
        for i in range(len(img_id)):
            raw_img_size = imgs[i].size
            target_cam = cam[i][target_cls[i]].unsqueeze(0).unsqueeze(0).detach().cpu().data
            up_target_cam = F.upsample(target_cam, size=(raw_img_size[1], raw_img_size[0]), mode='bilinear',
                                        align_corners=True)

            outlines = model.norm_cam_2_binary(up_target_cam.squeeze())
            max_cam = get_max_binary_area(outlines.numpy())
            bb = get_bbox_from_binary_cam(max_cam)
            bb2 = [float(st) for st in bbox[i].split(" ")]

            val_ious.append(get_iou(bb, bb2))

    # save points
    accs = np.array(val_result) == np.array(val_label)
    val_acc_arr.append(np.mean(accs))
    val_acc_iou.append(np.mean(np.array(accs) * np.array(val_ious)))

    save_results(args = args, result_dict = {
        'class_id': args.cls,
        'val_acc_arr': val_acc_arr,
        'iou': val_acc_iou
    })
    

def vgg_deconv2_cls():
    # 初始化
    torch.cuda.empty_cache()
    epoch = 0
    train_acc_arr = []
    val_acc_arr = []

    # 加载数据
    dataset = CUB_Loader(args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 加载模型
    model = get_vgg_deconv2_model(args=args)

    # 加载参数
    loss_func = torch.nn.CrossEntropyLoss()

    # 训练
    if args.continue_train:
        model.load_state_dict(torch.load(args.train_model))
        check_dict = load_check_point(args=args)
        epoch = check_dict['epoch']
        train_acc_arr = check_dict['train_acc_arr']
        val_acc_arr = check_dict['val_acc_arr']

    while epoch < args.epoch:
        opt = get_deconv_finetune_optimizer(args, model)

        train_result = []
        train_label = []
        val_result = []
        val_label = []

        for step, (img_id, img, label, bbox) in enumerate(dataloader):
            img = img.cuda()
            label = label.cuda()

            logits, logits2, cam, deconv_cam = model.forward(img)
            logits = logits * logits2
            loss = loss_func(logits, label)
            acc = cal_acc(logits, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print('epoch:{} train loss:{} train acc:{}'.format(epoch, loss, acc))

            train_result.extend(torch.argmax(logits, dim=-1).cpu().data.numpy())
            train_label.extend(label.cpu().data.numpy())

        train_acc_arr.append(np.mean(np.array(train_result) == np.array(train_label)))

        # validation
        dataset.to_val()
        val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
            img = img.cuda()
            label = label.cuda()

            logits, logits2, cam, deconv_cam = model.forward(img)
            logits = logits * logits2
            val_result.extend(torch.argmax(logits, dim=-1).cpu().data.numpy())
            val_label.extend(label.cpu().data.numpy())

            if step == 0:
                target_cls = torch.argmax(logits, dim=-1)

                plot_dict = dict()
                plot_dict['raw_imgs'] = get_raw_imgs_by_id(args, img_id[:5], dataset)
                target_cams = []
                target_deconv_cams = []
                for i in range(5):
                    raw_img_size = plot_dict['raw_imgs'][i].size
                    target_cam = cam[i][target_cls[i]].unsqueeze(0).unsqueeze(0).detach().cpu().data
                    up_target_cam = F.upsample(target_cam, size=(raw_img_size[1], raw_img_size[0]), mode='bilinear',
                                               align_corners=True)
                    target_cams.append(up_target_cam.squeeze())

                    target_deconv_cam = deconv_cam[i][target_cls[i]].unsqueeze(0).unsqueeze(0).detach().cpu().data
                    up_target_deconv_cam = F.upsample(target_deconv_cam, size=(raw_img_size[1], raw_img_size[0]),
                                                      mode='bilinear',
                                                      align_corners=True)
                    target_deconv_cams.append(up_target_deconv_cam.squeeze())

                plot_dict['cams'] = target_cams
                plot_dict['deconv_cam'] = target_deconv_cams
                plot_different_figs(args, plot_dict)

        val_acc_arr.append(np.mean(np.array(val_result) == np.array(val_label)))

        if len(val_acc_arr) == 1 or val_acc_arr[-1] >= val_acc_arr[-2]:
            torch.save(model.state_dict(), args.save_model_path)

        # plot
        plot_train_process(args, [train_acc_arr, val_acc_arr])

        # save check point
        epoch += 1
        save_check_point(args=args, check_dict={
            'epoch': epoch,
            'train_acc_arr': train_acc_arr,
            'val_acc_arr': val_acc_arr
        })

        dataset.to_train()


def vgg_auto_encoder():
    # 初始化
    torch.cuda.empty_cache()
    epoch = 0
    train_acc_arr = []
    val_acc_arr = []

    # 加载数据
    dataset = CUB_Loader(args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 加载模型
    model = get_vgg_auto_encoder_model(args=args)

    # 加载参数
    loss_func = torch.nn.CrossEntropyLoss()

    # 训练
    if args.continue_train:
        model.load_state_dict(torch.load(args.train_model))
        check_dict = load_check_point(args=args)
        epoch = check_dict['epoch']
        train_acc_arr = check_dict['train_acc_arr']
        val_acc_arr = check_dict['val_acc_arr']

    while epoch < args.epoch:
        opt = get_deconv_finetune_optimizer(args, model)

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

        train_acc_arr.append(np.mean(np.array(train_result) == np.array(train_label)))

        # validation
        dataset.to_val()
        val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
            img = img.cuda()
            label = label.cuda()

            logits, cam = model.forward(img)
            val_result.extend(torch.argmax(logits, dim=-1).cpu().data.numpy())
            val_label.extend(label.cpu().data.numpy())

            if step == 0:
                target_cls = torch.argmax(logits, dim=-1)

                plot_dict = dict()
                plot_dict['raw_imgs'] = get_raw_imgs_by_id(args, img_id[:5], dataset)
                target_cams = []
                for i in range(5):
                    raw_img_size = plot_dict['raw_imgs'][i].size
                    target_cam = cam[i][target_cls[i]].unsqueeze(0).unsqueeze(0).detach().cpu().data
                    up_target_cam = F.upsample(target_cam, size=(raw_img_size[1], raw_img_size[0]), mode='bilinear',
                                               align_corners=True)
                    target_cams.append(up_target_cam.squeeze())

                plot_dict['cams'] = target_cams
                plot_different_figs(args, plot_dict)

        val_acc_arr.append(np.mean(np.array(val_result) == np.array(val_label)))

        if len(val_acc_arr) == 1 or val_acc_arr[-1] >= val_acc_arr[-2]:
            torch.save(model.state_dict(), args.save_model_path)

        # plot
        plot_train_process(args, [train_acc_arr, val_acc_arr])

        # save check point
        epoch += 1
        save_check_point(args=args, check_dict={
            'epoch': epoch,
            'train_acc_arr': train_acc_arr,
            'val_acc_arr': val_acc_arr
        })

        dataset.to_train()


def graph_seg_bbox():
    dataset = CUB_Loader(args=args, mode='val')
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
        data_dict = dict()
        data_dict['img_path'] = get_img_path_by_id(args, img_id, dataset)
        data_dict['raw_imgs'] = get_raw_imgs_by_id(args, img_id, dataset)
        data_dict['img_id']=img_id
        get_seg_by_path(args=args,data_dict=data_dict)

def dilation_vgg():
    # 初始化
    torch.cuda.empty_cache()
    epoch = 0
    train_acc_arr = []
    val_acc_arr = []

    # 加载数据
    dataset = CUB_Loader(args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 加载模型
    model = get_dilation_vgg_model(args=args)

    # 加载参数
    loss_func = torch.nn.CrossEntropyLoss()

    # 训练
    if args.continue_train:
        model.load_state_dict(torch.load(args.train_model))
        check_dict = load_check_point(args=args)
        epoch = check_dict['epoch']
        train_acc_arr = check_dict['train_acc_arr']
        val_acc_arr = check_dict['val_acc_arr']

    while epoch < args.epoch:
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

        train_acc_arr.append(np.mean(np.array(train_result) == np.array(train_label)))

        # validation
        dataset.to_val()
        val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for step, (img_id, img, label, bbox) in enumerate(tqdm(val_dataloader)):
            img = img.cuda()
            label = label.cuda()

            logits, cam = model.forward(img)
            val_result.extend(torch.argmax(logits, dim=-1).cpu().data.numpy())
            val_label.extend(label.cpu().data.numpy())

            if step == 0:
                target_cls = torch.argmax(logits, dim=-1)

                plot_dict = dict()
                plot_dict['raw_imgs'] = get_raw_imgs_by_id(args, img_id[:5], dataset)
                target_cams = []
                for i in range(5):
                    raw_img_size = plot_dict['raw_imgs'][i].size
                    target_cam = cam[i][target_cls[i]].unsqueeze(0).unsqueeze(0).detach().cpu().data
                    up_target_cam = F.upsample(target_cam, size=(raw_img_size[1], raw_img_size[0]), mode='bilinear',
                                               align_corners=True)
                    target_cams.append(up_target_cam.squeeze())

                plot_dict['cams'] = target_cams
                plot_different_figs(args, plot_dict)

        val_acc_arr.append(np.mean(np.array(val_result) == np.array(val_label)))

        if len(val_acc_arr) == 1 or val_acc_arr[-1] >= val_acc_arr[-2]:
            torch.save(model.state_dict(), args.save_model_path)

        # plot
        plot_train_process(args, [train_acc_arr, val_acc_arr])

        # save check point
        epoch += 1
        save_check_point(args=args, check_dict={
            'epoch': epoch,
            'train_acc_arr': train_acc_arr,
            'val_acc_arr': val_acc_arr
        })

        dataset.to_train()


if __name__ == '__main__':
    args = parse_args()
    for cls_id in range(args.tot_cls):
        print("current cls: {}".format(cls_id))
        args.cls = cls_id
        model, dataset = base_vgg_cls()
        base_vgg_bbx(model, dataset)
        del model
        del dataset
        gc.collect()