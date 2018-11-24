import pandas as pd
import numpy as np
import ast
import os

ROOT_DIR = '../../../../../../../Data/CUB2011/CUB_200_2011/'


def main():
    data = []

    with open(ROOT_DIR + 'images.txt') as f:
        lines = f.readlines()
        for line in lines:
            image_name, info = line.split(' ')
            class_info, img_path = info.split('/')
            class_info = class_info[:3]

            data.append([image_name, info.strip(), int(class_info)])

    # init csv image_name,path,label
    data_csv = pd.DataFrame(data, columns=['img_name', 'path', 'label'])

    # get is_train
    data_csv['is_train'] = np.nan
    is_train_map = {}
    with open(ROOT_DIR + 'train_test_split.txt') as f:
        lines = f.readlines()
        for line in lines:
            img_name, bool_val = line.split(' ')
            is_train_map[img_name] = int(bool_val.strip())
    for index in data_csv.index:
        img_name = data_csv.at[index, 'img_name']
        if img_name in is_train_map.keys():
            data_csv.at[index, 'is_train'] = int(is_train_map[img_name])

    tmp = data_csv.head()

    # get bbox
    data_csv['bbox'] = ''
    bbox_map = {}
    with open(ROOT_DIR + 'bounding_boxes.txt') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            bbox_map[items[0]] = str([int(float(item)) for item in items[1:]])
    for index in data_csv.index:
        img_name = data_csv.at[index, 'img_name']
        if img_name in bbox_map.keys():
            data_csv.at[index, 'bbox'] = bbox_map[img_name]

    data_csv.to_csv('../../Save/CUB2011/data.csv', index=False)


if __name__ == '__main__':
    main()
