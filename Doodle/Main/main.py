import sys

sys.path.append('../')

from torch.utils.data import DataLoader
from dataloader.MIL_Custom_Dataset import Doodle
from model.simple_mil import Simple_Classify_MIL
import argparse
from Main.custom_optim import *
from functions.metrics import *
import pandas as pd
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--data_root_path', type=str, default='../../../../../Data/Doodle/')

    parser.add_argument('--lr', type=float, default=0.0001)

    args = parser.parse_args()
    return args


def doodle_model():
    dataset = Doodle(train_csv_path='../Save/train_data.csv', val_csv_path='../Save/val_data.csv',
                     test_csv_path='../Save/test_data.csv', args=args, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = Doodle(train_csv_path='../Save/train_data.csv', val_csv_path='../Save/val_data.csv',
                         test_csv_path='../Save/test_data.csv', args=args, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = Simple_Classify_MIL()
    model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()

    best_test_acc = 0.0
    for epoch in range(args.epoch):
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        cur_opt = decrease_lr_by_epoch(epoch, model, args)
        for step, (b_name, b_x, b_y) in enumerate(dataloader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()

            logits, A = model.forward(b_x)
            loss = loss_func(logits, b_y)
            acc = cal_single_label_acc(logits, b_y)

            cur_opt.zero_grad()
            loss.backward()
            cur_opt.step()

            train_loss += loss.item()
            train_acc += acc

            print('train loss: '+str(loss)+' | train acc: '+str(acc))

        with torch.no_grad():
            for step, (b_name, b_x, b_y) in enumerate(val_dataloader):
                b_x = b_x.cuda()
                b_y = b_y.cuda()

                logits, A = model.forward(b_x)
                loss = loss_func(logits, b_y)
                acc = cal_single_label_acc(logits, b_y)

                test_loss += loss
                test_acc += acc

        print('epoch: ' + str(epoch) + ' | train loss: ' + str(train_loss) + ' | train acc: ' + str(
            train_acc / len(dataloader)) +
              ' | test loss: ' + str(test_loss) + ' | test acc: ' + str(test_acc / len(val_dataloader)))

        if test_acc / len(val_dataloader) > best_test_acc:
            best_test_acc = test_acc / len(val_dataloader)
            torch.save(model, '../Save/model/mil_model.pt')


def doodle_test():
    dataset = Doodle(train_csv_path='../Save/train_data.csv', val_csv_path='../Save/val_data.csv',
                     test_csv_path='../Save/test_data.csv', args=args, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = torch.load('../Save/model/mil_model.pt')
    model.cuda()

    word_list = np.array(pd.read_csv(args.data_root_path + 'train/word_list.csv')['word'])

    result_arr=[]
    result_map={}
    for step, (b_name, b_x) in enumerate(dataloader):
        b_x = b_x.cuda()
        if step % 100 == 0:
            print(step)

        logits, A = model.forward(b_x)
        result_arr.append([b_name[0],' '.join(word_list[np.argsort(logits.squeeze().cpu().data.numpy())[-3:]][::-1])])
        # result_arr.append([str(b_name[0].cpu().data.numpy()),])
        # result_map[str(b_name[0].cpu().data.numpy())] = np.argsort(np_result)[-3:]

    pd.DataFrame(data=result_arr, columns=['key_id', 'word']).to_csv('../Save/result/submission.csv', index=False)
    print()


if __name__ == '__main__':
    args = parse_args()
    doodle_test()
