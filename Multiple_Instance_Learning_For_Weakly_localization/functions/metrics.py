import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image, ImageDraw
from scipy import ndimage
import ast
import matplotlib

'''
单分类问题acc
'''


def cal_single_label_acc(result, label, top_k=1):
    label = label.cpu().data.numpy()

    if top_k > 1:
        max_index = np.argsort(result.cpu().data.numpy(), axis=-1)[-1 * top_k:]

        return np.sum(max_index == label) / float(len(label))
    else:
        max_index = np.argmax(result.squeeze().cpu().data.numpy(), axis=-1)

        return np.sum(max_index == label) / float(len(label))


def cal_sigmoid_acc(result, label):
    label = label.cpu().data.numpy()
    result = result.squeeze().cpu().data.numpy()

    return np.sum(result == label) / len(result)


def mil_merge_to_one(logits):
    max_index = np.argmax(logits.cpu().data.numpy(), axis=-1)

    count = np.argmax(np.bincount(max_index))
    count_index = np.where(max_index == count)[0]
    return count, count_index


def check_x_to_y(img_name, b_y, args):
    train_csv = pd.read_csv(args.train_data_path)
    img_path = list(train_csv[train_csv['img_name'].isin(img_name)]['path'])
    true_name = np.array([int(item.split('.')[0]) for item in img_path]) - 1
    return set(true_name) == set(b_y.data.numpy())


def cal_whole_loss(logits, b_y, loss_func):
    total_loss = 0.0
    for logit in logits:
        total_loss += loss_func(logit, b_y)

    return total_loss / len(logits)


def normalize_atten_maps(atten_maps):
    atten_shape = atten_maps.size()

    # --------------------------
    batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
    batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
    atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins,
                             batch_maxs - batch_mins)
    atten_normed = atten_normed.view(atten_shape)

    return atten_normed


def outline_peak_maps(atten_maps, win_size=3):
    atten_shape = atten_maps.size()

    offset = (win_size - 1) // 2
    padding = torch.nn.ConstantPad2d(offset, float('-inf'))
    padded_maps = padding(atten_maps)
    batch_size, num_channels, h, w = padded_maps.size()
    element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
    element_map = element_map.to(atten_maps.device)
    _, indices = F.max_pool2d(
        padded_maps,
        kernel_size=win_size,
        stride=1,
        return_indices=True)
    peak_map = (indices == element_map)

    return peak_map


def outline_atten_maps(atten_maps, win_size=3):
    atten_shape = atten_maps.size()

    offset = (win_size - 1) // 2
    padding = torch.nn.ConstantPad2d(offset, float('-inf'))
    padded_maps = padding(atten_maps)
    batch_size, num_channels, h, w = padded_maps.size()
    element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
    element_map = element_map.to(atten_maps.device)
    _, indices = F.max_pool2d(
        padded_maps,
        kernel_size=win_size,
        stride=1,
        return_indices=True)
    peak_map = (indices == element_map)

    return peak_map


def outline_to_large_area(outline):
    conv2d_kernel = np.ones((3, 3))
    conv2d_kernel[1, 1] = 0
    np_outline = outline.data.numpy()
    conv2d = signal.convolve2d(np_outline, conv2d_kernel, mode='same')

    np_outline = np.zeros(conv2d.shape)
    np_outline[conv2d > 4] = 1.0

    conv2d_kernel = np.ones((5, 5))
    conv2d_kernel[2, 2] = 0
    conv2d = signal.convolve2d(np_outline, conv2d_kernel, mode='same')

    np_outline = np.zeros(conv2d.shape)
    np_outline[conv2d > 4] = 1.0

    conv2d_kernel = np.ones((7, 7))
    conv2d_kernel[3, 3] = 0
    conv2d = signal.convolve2d(np_outline, conv2d_kernel, mode='same')

    np_outline = np.zeros(conv2d.shape)
    np_outline[conv2d > 4] = 1.0

    return np_outline


def plot_raw_cam_bi(raw, draw_raw, cam, norm_cam_np, bs=False, mode='plot'):
    if mode == 'plot':
        if raw is not None:
            plt.imshow(raw)
            plt.show()

        if draw_raw is not None:
            plt.imshow(draw_raw)
            plt.show()

        if cam is not None:
            plt.imshow(cam[0].squeeze().cpu().data.numpy(), cmap='hot')
            plt.show()

        if norm_cam_np is not None:
            plt.imshow(norm_cam_np)
            plt.show()

    else:
        if raw is not None:
            plt.imshow(raw)
            plt.savefig('../Save/images/raw.png')

        if draw_raw is not None:
            if bs:
                plt.imshow(draw_raw)
                plt.savefig('../Save/images/draw_raw_baseline.png')
            else:
                plt.imshow(draw_raw)
                plt.savefig('../Save/images/draw_raw.png')

        if cam is not None:
            plt.imshow(cam[0].squeeze().cpu().data.numpy() - 0.2, cmap='hot')
            plt.savefig('../Save/images/cam.png')

        if norm_cam_np is not None:
            plt.imshow(norm_cam_np)
            plt.savefig('../Save/images/norm_cam_np.png')


def get_raw_imgs(b_name, dataset):
    img_path = '../../../../../Data/CUB2011/CUB_200_2011/images/' + \
               dataset.test_data[dataset.test_data['img_name'] == int(b_name[0])]['path'].values[0]
    raw_img = Image.open(img_path)
    bbox = dataset.test_data[dataset.test_data['img_name'] == int(b_name[0])]['bbox'].values[0]

    return img_path, raw_img, bbox


def get_raw_imgnet_imgs(b_name, dataset):
    img_path = dataset.val_data_csv[dataset.val_data_csv['ImageId'] == b_name[0]]['path'].values[0]
    raw_img = Image.open(img_path)
    bbox = dataset.val_data_csv[dataset.val_data_csv['ImageId'] == b_name[0]]['bbox'].values[0]

    return img_path, raw_img, bbox


def cal_iou(norm_cam_np, bbox, raw, mode='cub'):
    if mode == 'imgnet':
        bbox = [int(x) for x in bbox.split(' ')]
    else:
        bbox = ast.literal_eval(bbox)

    draw = ImageDraw.Draw(raw)
    if mode == 'imgnet':
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline='green')
    else:
        draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline='green')

    # points = np.where(norm_cam_np == 1)
    # bbox_points = [points[1].min(), points[0].min(), points[1].max(), points[0].max()]
    # draw.rectangle(bbox_points, outline='red')
    #
    # iou = get_iou(norm_cam_np, bbox)

    iou = 0
    return raw, iou


def get_max_binary_area(norm_cam_np):
    if len(norm_cam_np.shape) == 3:
        cams = np.ones(norm_cam_np.shape)
        for i in range(norm_cam_np.shape[0]):
            label, num_label = ndimage.label(norm_cam_np[i] == 1)
            size = np.bincount(label.ravel())
            biggest_label = size[1:].argmax() + 1
            clump_mask = label == biggest_label

            largest_norm_cam = norm_cam_np[i] * clump_mask
            cams[i] = largest_norm_cam

        return cams
    else:
        label, num_label = ndimage.label(norm_cam_np == 1)
        size = np.bincount(label.ravel())
        biggest_label = size[1:].argmax() + 1
        clump_mask = label == biggest_label

        largest_norm_cam = norm_cam_np * clump_mask

        return largest_norm_cam


def is_larger(norm_cam_np, bbox):
    if bbox.__class__.__name__ == 'Tensor':
        bbox = bbox.numpy()

    if type(bbox) is str:
        bbox = ast.literal_eval(bbox)

    points = np.where(norm_cam_np == 1)
    bbox_points = [points[1].min(), points[0].min(), points[1].max(), points[0].max()]

    bb1 = {'x1': bbox_points[0], 'x2': bbox_points[2], 'y1': bbox_points[1], 'y2': bbox_points[3]}
    bb2 = {'x1': bbox[0], 'x2': bbox[0] + bbox[2], 'y1': bbox[1],
           'y2': bbox[1] + bbox[3]}

    if bb1['x1'] <= bb2['x1'] and bb1['y1'] <= bb2['y1'] and bb1['x2'] >= bb2['x2'] and bb1['y2'] >= bb2['y2']:
        return True
    else:
        return False


def get_iou(norm_cam_np, bbox):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    if bbox.__class__.__name__ == 'Tensor':
        bbox = bbox.numpy()

    points = np.where(norm_cam_np == 1)
    bbox_points = [points[1].min(), points[0].min(), points[1].max(), points[0].max()]
    bb1 = {'x1': bbox_points[0], 'x2': bbox_points[2], 'y1': bbox_points[1], 'y2': bbox_points[3]}
    bb2 = {'x1': bbox[0], 'x2': bbox[0] + bbox[2], 'y1': bbox[1],
           'y2': bbox[1] + bbox[3]}

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
