import os
import numpy as np
import pickle
from PIL import Image
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from xml.dom import minidom
from tqdm import tqdm

cotegory = ['dog', 'sofa', 'bicycle', 'aeroplane', 'pottedplant', 'car', 'horse', 'bird', 'bottle', 'cow', 'cat',
            'tvmonitor', 'person', 'train', 'boat', 'diningtable', 'sheep', 'chair', 'motorbike', 'bus']

original_img_dir = '../../../../../../../Data/VOCdevkit/VOC2012/JPEGImages/'
original_anno_dir = '../../../../../../../Data/VOCdevkit/VOC2012/Annotations/'


def get_original_img_h_w(img_name):
    img = Image.open(original_img_dir + img_name + '.jpg')

    return img.size


def get_target_obj_bbox(img_name, obj_index):
    configure_doc = minidom.parse(original_anno_dir + img_name + '.xml')
    root = configure_doc.documentElement
    objects = root.getElementsByTagName('object')

    target_bbox = []
    target_label = cotegory[obj_index]
    for i, object in enumerate(objects):
        name = object.getElementsByTagName('name')[0].firstChild.data
        if name == target_label:
            bndbox = object.getElementsByTagName('bndbox')[0]

            # [x_min,y_min,x_max,y_max]
            single_box = []
            single_box.append([child.firstChild.data for child in bndbox.childNodes if
                               child.nodeType == 1 and child.nodeName == 'xmin'][0])
            single_box.append([child.firstChild.data for child in bndbox.childNodes if
                               child.nodeType == 1 and child.nodeName == 'ymin'][0])
            single_box.append([child.firstChild.data for child in bndbox.childNodes if
                               child.nodeType == 1 and child.nodeName == 'xmax'][0])
            single_box.append([child.firstChild.data for child in bndbox.childNodes if
                               child.nodeType == 1 and child.nodeName == 'ymax'][0])

            target_bbox.append(single_box)

    return target_bbox


def is_in_bbox(row, col, bbox):
    if col > int(bbox[0]) and col < int(bbox[2]) and row > int(bbox[1]) and row < int(bbox[3]):
        return True
    else:
        return False


def cal_mAP(results_path):
    result_map = {}
    for cate in cotegory:
        result_map[cate] = []

    files = [file for file in os.listdir(results_path) if 'pkl' in file]
    for index, file in tqdm(enumerate(files)):
        with open(results_path + file, 'rb') as f:
            img_name, img_labels, aggregation, class_response_maps, whole_pickle_list_in_target_class, valid_peak_list, peak_response_maps = pickle.load(
                f)
            aggregation_label = [1 if aggr > 0 else 0 for aggr in aggregation]
            for img_label_index, img_label in enumerate(img_labels):
                if aggregation_label[img_label_index] == 0 and img_labels[img_label_index] == 0:
                    result_map[cotegory[img_label_index]].append([0, 0])
                elif aggregation_label[img_label_index] == 0 and img_labels[img_label_index] == 1:
                    result_map[cotegory[img_label_index]].append([0, 1])
                elif aggregation_label[img_label_index] == 1 and img_labels[img_label_index] == 0:
                    result_map[cotegory[img_label_index]].append([1, 0])
                else:
                    indexed_response_map = class_response_maps[img_label_index]
                    img_w_h = get_original_img_h_w(img_name)
                    upsample_indexed_response_map = misc.imresize(indexed_response_map, (img_w_h[1], img_w_h[0]))
                    max_response_position = np.where(
                        upsample_indexed_response_map == np.max(upsample_indexed_response_map))

                    # plt.imshow(upsample_indexed_response_map)
                    # plt.show()

                    isLocated = False
                    rows = max_response_position[0]
                    cols = max_response_position[1]
                    for pair in zip(rows, cols):
                        row = pair[0]
                        col = pair[1]
                        for bbox in get_target_obj_bbox(img_name, img_label_index):
                            if is_in_bbox(row, col, bbox):
                                isLocated = True
                                break

                        if isLocated:
                            break

                    if isLocated:
                        result_map[cotegory[img_label_index]].append([1, 1])
                    else:
                        result_map[cotegory[img_label_index]].append([0, 1])

    print()


if __name__ == '__main__':
    cal_mAP('results/')
