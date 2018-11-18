import sys

sys.path.append('../')

from data.customDataSet import VOC, image_transform, DataLoader
from model.peak_response_map import peak_response_mapping
from model.resnet_50 import fc_resnet50
from torch.nn import MultiLabelSoftMarginLoss
import torch
import numpy as np
from tqdm import tqdm
import pickle

BATCH_SIZE = 1
EPOCH = 20


def cal_acc(results, labels):
    acc_num = 0
    acc_total_num = 0
    for i in range(len(results)):
        if results[i] > 0 and labels[i] == 1:
            acc_num += 1
            acc_total_num += 1
        elif results[i] > 0 and labels[i] == 0 or results[i] <= 0 and labels[i] == 1:
            acc_total_num += 1

    return float(acc_num) / acc_total_num


def main():
    # val_dataset = VOC(mode='val', img_transform=image_transform())
    # val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    obj_seg_dataset = VOC(mode='ins_seg', img_transform=image_transform(mode='seg'))
    obj_seg_dataloader = DataLoader(obj_seg_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = torch.load('../Save/model/model.pt')
    model.cuda()

    # with torch.no_grad():
    #     val_results = []
    #     val_labels = []
    #     for step, (b_x, b_y) in enumerate(val_dataloader):
    #         b_x = b_x.cuda()
    #         b_y = b_y.cuda()
    #         result = model.forward(b_x)
    #
    #         val_results.extend(result.cpu().data.numpy())
    #         val_labels.extend(b_y.cpu().data.numpy())
    #
    #     val_results = np.array(val_results).reshape(-1)
    #     val_labels = np.array(val_labels).reshape(-1)
    #
    #     print(' val_acc:' + str(cal_acc(val_results, val_labels)))

    model.inference()
    with torch.enable_grad():
        for step, (b_name, b_x, b_y) in enumerate(obj_seg_dataloader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()

            aggregation, class_response_maps, peak_list, valid_peak_list, peak_response_maps = model.forward(b_x)

            with open('../Save/data/peak_response_maps/results/' + b_name[0] + '.pkl', 'wb') as f:
                pickle.dump(
                    [b_name[0], torch.squeeze(b_y).cpu().data.numpy(), torch.squeeze(aggregation).cpu().data.numpy(),
                     torch.squeeze(class_response_maps).cpu().data.numpy(), peak_list.cpu().data.numpy(),
                     valid_peak_list.cpu().data.numpy(), peak_response_maps.cpu().data.numpy()], f)

    print('peak successfully')


if __name__ == '__main__':
    main()
