import cv2
import scipy.io as sio
import numpy as np
from os import listdir
import os

for mode in ['train', 'test']:
    vid_path = '../../../../Data/PennAction/frames/'
    ann_path = '../../../../Data/PennAction/labels/'
    pad = 5
    f = open('../../../../Data/PennAction/' + mode + '_list.txt', 'r')
    lines = f.readlines()
    f.close()
    numvids = len(lines)

    for i, line in enumerate(lines):
        tokens = line.split()[0].split('frames')
        ff = sio.loadmat('../../../../Data/PennAction/labels/' + tokens[1] + '.mat')
        bboxes = ff['bbox']
        posey = ff['y']
        posex = ff['x']
        visib = ff['visibility']
        imgs = sorted(
            [f for f in listdir('../../../.' + line.split()[0].replace('datasets', 'Data')) if f.endswith('.jpg')])

        box = np.zeros((4,), dtype='int32')
        bboxes = bboxes.round().astype('int32')

        if len(imgs) > bboxes.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes[-1][None]), axis=0)

        box[0] = bboxes[:, 0].min()
        box[1] = bboxes[:, 1].min()
        box[2] = bboxes[:, 2].max()
        box[3] = bboxes[:, 3].max()

        for j in range(len(imgs)):
            img = cv2.imread('../../../.' + line.split()[0].replace('datasets', 'Data') + '/' + imgs[j])
            x1 = box[0] - pad
            y1 = box[1] - pad
            x2 = box[2] + pad
            y2 = box[3] + pad

            h = y2 - y1 + 1
            w = x2 - x1 + 1

            if h > w:
                left_pad = (h - w) / 2
                right_pad = (h - w) / 2 + (h - w) % 2
                x1 -= left_pad
                if x1 < 0:
                    x1 = 0
                x2 += right_pad
                if x2 > img.shape[1]:
                    x2 = img.shape[1]

            elif w > h:
                up_pad = (w - h) / 2
                down_pad = (w - h) / 2 + (w - h) % 2
                y1-=up_pad
                if y1<0:
                    y1=0
                y2+=down_pad
                if y2>img.shape[0]:
                    y2=img.shape[0]
