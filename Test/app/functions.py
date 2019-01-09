import numpy as np
import torch
import shutil
import torch.nn.functional as F
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw
from scipy import ndimage


def cal_acc(logits, label):
    pre = torch.argmax(logits, dim=-1)

    return torch.mean((pre == label).type(torch.FloatTensor)).item()


def save_draw_imgs_by_id(args,img_id, logits, cams, dataset):
    data = dataset.cur_csv

    for index, id in enumerate(img_id):
        path = data[data['ImageId'] == id]['path'].values[0]
        gt_bbox = data[data['ImageId'] == id]['bbox'].values[0]

        # shutil.copyfile(args.data_root_path + path, '../save/train/imgs/{}.JPEG'.format(id))

        target_cam = cams[index].unsqueeze(0)
        raw_img = PIL.Image.open(args.data_root_path + path).convert('RGB')
        up_target_cam = F.upsample(target_cam, size=(raw_img.size[1], raw_img.size[0]), mode='bilinear',
                                   align_corners=True).squeeze()
        cam = norm_cam_2_binary(up_target_cam[torch.argmax(logits[index], dim=-1)])

        if len(gt_bbox.split(' ')) == 4:
            iou, bbox, gt_bbox = cal_iou(cam, [float(x) for x in gt_bbox.split(' ')])
            draw = ImageDraw.Draw(raw_img)
            draw.rectangle([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']], outline='red')
            draw.rectangle([gt_bbox['x1'], gt_bbox['y1'], gt_bbox['x2'], gt_bbox['y2']], outline='green')

        plt.imshow(raw_img)
        plt.savefig('../save/train/imgs/{}.JPEG'.format(id))
        
def save_each_layer_cams_by_id(args,model,dataset,img_id):
    data = dataset.cur_csv
    
    for index, id in enumerate(img_id):
        path = data[data['ImageId'] == id]['path'].values[0]
        gt_bbox = data[data['ImageId'] == id]['bbox'].values[0]
        
        target_cams=[]
        for cam in model.each_layer_feature_map:
            target_cams.append(cam[index])

def norm_cam_2_binary(cam):
    grad_shape = cam.size()
    outline = torch.zeros(grad_shape)

    thd = float(np.percentile(np.sort(cam.view(-1).cpu().data.numpy()), 80))
    high_pos = torch.gt(cam, thd)
    outline[high_pos.data] = 1.0

    cam_np = outline.cpu().data.numpy()
    label, num_label = ndimage.label(cam_np == 1)
    size = np.bincount(label.ravel())
    biggest_label = size[1:].argmax() + 1
    clump_mask = label == biggest_label

    largest_norm_cam = cam_np * clump_mask

    return largest_norm_cam


def cal_iou(cam, gt_bbox):
    points = np.where(cam == 1)
    bbox_points = [points[1].min(), points[0].min(), points[1].max(), points[0].max()]

    bb1 = {'x1': bbox_points[0], 'x2': bbox_points[2], 'y1': bbox_points[1], 'y2': bbox_points[3]}
    bb2 = {'x1': gt_bbox[0], 'x2': gt_bbox[2], 'y1': gt_bbox[1],
           'y2': gt_bbox[3]}

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
    return iou, bb1, bb2
