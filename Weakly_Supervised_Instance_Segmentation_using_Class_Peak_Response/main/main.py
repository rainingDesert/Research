import sys

sys.path.append('../')

from data.customDataSet import VOC, image_transform, DataLoader
from model.peak_response_map import peak_response_mapping
from model.resnet_50 import fc_resnet50
from model.linear_transform import function_linear_transform
from torch.nn import MultiLabelSoftMarginLoss
import torch
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 16
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
    dataset = VOC(img_transform=image_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = VOC(mode='val', img_transform=image_transform())
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = peak_response_mapping(fc_resnet50(), win_size=3, sub_pixel_locating_factor=8,
                                  enable_peak_stimulation=True)
    model = model.cuda()

    loss_func = MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1.0e-4)

    val_acc = 0
    for epoch in range(EPOCH):
        print('-----------epoch' + str(epoch) + '-----------')
        for step, (b_x, b_y) in enumerate(dataloader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()

            result = model.forward(b_x)
            loss = loss_func(result, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss:' + str(loss.cpu().data.numpy()) + ' acc:' + str(
                cal_acc(result.cpu().data.numpy().reshape(-1), b_y.cpu().data.numpy().reshape(-1))))

            if step % 10 == 0 and step != 0:
                with torch.no_grad():
                    val_loss = []
                    val_results = []
                    val_labels = []
                    for step, (b_x, b_y) in enumerate(val_dataloader):
                        b_x = b_x.cuda()
                        b_y = b_y.cuda()
                        result = model.forward(b_x)
                        single_loss = loss_func(result, b_y)
                        val_loss.append(single_loss.cpu().data.numpy())
                        optimizer.zero_grad()

                        val_results.extend(result.cpu().data.numpy())
                        val_labels.extend(b_y.cpu().data.numpy())

                    val_results = np.array(val_results).reshape(-1)
                    val_labels = np.array(val_labels).reshape(-1)

                    print('--------------------val_loss:' + str(np.mean(val_loss)) + ' val_acc:' + str(
                        cal_acc(val_results, val_labels)))

                    if (cal_acc(val_results, val_labels) > val_acc):
                        val_acc = cal_acc(val_results, val_labels)
                        torch.save(model, '../Save/model/model.pt')


if __name__ == '__main__':
    main()
