import torch
import pickle
from PIL import Image
from scipy import ndimage
import numpy as np

def cal_acc(logits,labels):
    return torch.mean((torch.argmax(logits,dim=-1)==labels).float()).item()

def save_check_point(args,check_dict):
    with open(args.check_path,'wb') as f:
        pickle.dump(check_dict,f)

def load_check_point(args):
    with open(args.check_path,'rb') as f:
        return pickle.load(f)
        
def read_one_fig(path):
    return Image.open(path)

def get_raw_imgs_by_id(args,img_id,dataset):
    raw_imgs=[]
    
    cur_csv=dataset.cur_csv
    target_path=cur_csv[cur_csv['id'].isin(img_id)]['path']
    for path in target_path:
        raw_imgs.append(Image.open(args.data_root_path + path))
        
    return raw_imgs

def get_bbox_from_binary_cam(bi_cam):
    points = np.where(bi_cam == 1)
    bbox_points = [points[1].min(), points[0].min(), points[1].max(), points[0].max()]

    bb1 = {'x1': bbox_points[0], 'x2': bbox_points[2], 'y1': bbox_points[1], 'y2': bbox_points[3]}
    
    return bb1

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

def get_iou(bb1, gt_bbx):
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
    bb2 = {'x1': gt_bbx[0], 'x2': gt_bbx[0] + gt_bbx[2], 'y1': gt_bbx[1],
           'y2': gt_bbx[1] + gt_bbx[3]}

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

    