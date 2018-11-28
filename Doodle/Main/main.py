import sys
sys.path.append('../')

from torch.utils.data import DataLoader
from dataloader.Custom_Dataset import Doodle
from model.vgg import get_vgg
import argparse
from Main.custom_optim import *
from functions.metrics import *
import pandas as pd
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--data_root_path', type=str, default='../../../../../Data/Doodle/')

    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    return args


def doodle_model():
    dataset = Doodle(train_csv_path='../Save/train_data.csv', val_csv_path='../Save/val_data.csv',
                     test_csv_path='../Save/test_data.csv', args=args, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = Doodle(train_csv_path='../Save/train_data.csv', val_csv_path='../Save/val_data.csv',
                         test_csv_path='../Save/test_data.csv', args=args, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size * 10, shuffle=False)

    model = get_vgg(pretrained=True, num_classes=340, is_RGB=False)
    model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    best_test_acc = 0.0
    for epoch in range(args.epoch):
        cur_opt = decrease_lr_by_epoch(epoch, model, args)
        for step, (b_name, b_x, b_y) in enumerate(dataloader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()

            cam, logits = model.forward(b_x)
            loss = loss_func(logits, b_y)
            acc = cal_single_label_acc(logits, b_y)

            cur_opt.zero_grad()
            loss.backward()
            cur_opt.step()

            train_loss.append(loss.cpu().data.numpy())
            train_acc.append(acc)

            print('mode: train | ' + 'epoch ' + str(epoch) + ' | ' + 'loss: ' + str(
                loss.cpu().data.numpy()) + ' | acc: ' + str(acc))

            if step % 50 == 0:
                with torch.no_grad():
                    for step, (b_name, b_x,b_y) in enumerate(val_dataloader):
                        b_x = b_x.cuda()
                        b_y = b_y.cuda()

                        cam, logits = model.forward(b_x)
                        loss = loss_func(logits, b_y)
                        acc = cal_single_label_acc(logits, b_y)

                        print(
                            'mode: val | ' + 'epoch ' + str(epoch) + ' | ' + 'loss: ' + str(
                                loss.cpu().data.numpy()) + ' | acc:' + str(
                                acc))

                        test_loss.append(loss.cpu().data.numpy())
                        test_acc.append(acc)

                        if acc > best_test_acc:
                            best_test_acc = acc
                            torch.save(model, '../Save/model/model.pt')

        with open('../Save/train.pkl', 'wb') as f:
            pickle.dump([train_loss, train_acc, test_loss, test_acc], f)


def doodle_test():
    dataset = Doodle(train_csv_path='../Save/train_data.csv', val_csv_path='../Save/val_data.csv',
                     test_csv_path='../Save/test_data.csv', args=args, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = torch.load('../Save/model/model.pt')
    model.cuda()

    word_list = np.array(pd.read_csv(args.data_root_path + 'train/word_list.csv')['word'])

    result_arr = []
    for step, (b_name, b_x) in enumerate(dataloader):
        b_x = b_x.cuda()
        if step%100==0:
            print(step)

        cam, logits = model.forward(b_x)
        np_result = logits.cpu().data.numpy()[0]
        result_arr.append([b_name[0], ' '.join(list(word_list[np.argsort(np_result)[-3:]]))])
        # result_arr.append([str(b_name[0].cpu().data.numpy()),])
        # result_map[str(b_name[0].cpu().data.numpy())] = np.argsort(np_result)[-3:]

    pd.DataFrame(data=result_arr, columns=['key_id', 'word']).to_csv('../Save/result/submission.csv', index=False)


if __name__ == '__main__':
    args = parse_args()
    doodle_test()
