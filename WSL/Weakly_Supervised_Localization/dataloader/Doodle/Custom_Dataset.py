from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from xml.dom import minidom
from torchvision import transforms
import numpy as np
import torch
import os


class Doodle(Dataset):
    def __init__(self, train_csv_path, val_csv_path, test_csv_path, args, mode='train', val_frac=0.00003):
        self.args = args
        self.mode = mode
        self.train_data = pd.read_csv(train_csv_path)
        self.val_data = pd.read_csv(val_csv_path)
        self.test_data = pd.read_csv(test_csv_path)

        print()

        # self.test_data.rename(columns={self.test_data.columns[0]:'img_name'},inplace=True)
        # self.test_data.to_csv('../Save/Doodle/test_list.csv',index=False)
        # print()
        # self.train_data.rename(columns={self.train_data.columns.values[0]:'img_name'},inplace=True)
        # self.val_data=self.train_data.sample(frac=val_frac)
        #
        # self.train_data.drop(index=self.val_data.index,inplace=True)
        #
        # self.train_data.reset_index(inplace=True)
        # self.val_data.reset_index(inplace=True)
        #
        # self.train_data.to_csv('../Save/Doodle/train_list.csv',index=False)
        # self.val_data.to_csv('../Save/Doodle/val_list.csv',index=False)

    def __getitem__(self, index):
        if self.mode == 'train':
            item = self.train_data.loc[index]

            img_name = item['img_name']
            img_path = item['Path']
            img_label = torch.squeeze(torch.from_numpy(np.array(int(str(item['Label']).strip())).reshape(-1)))

            img = Image.open(
                self.args.data_root_path + 'train/simplified/' + img_path.split('/')[-1].split('.')[0] + '/' +
                img_path.split('/')[-1])
            img = image_transform(mode='train')(img)

            return img_name, img, img_label

        elif self.mode == 'val':
            item = self.val_data.loc[index]
            img_name = item['img_name']
            img_path = item['Path']
            img_label = torch.squeeze(torch.from_numpy(np.array(int(str(item['Label']).strip())).reshape(-1)))
            img = Image.open(
                self.args.data_root_path + 'train/simplified/' + img_path.split('/')[-1].split('.')[0] + '/' +
                img_path.split('/')[-1])
            img = image_transform(mode='val')(img)

            return img_name, img, img_label

        else:
            item = self.test_data.loc[index]
            img_name = item['img_name']
            img_path = item['path']
            img = Image.open(
                self.args.data_root_path + 'test/images/' + img_path.split('/')[-1] + '/' +
                img_path.split('/')[-1]+'.png')
            img = image_transform(mode='test')(img)

            return img_path.split('/')[-1], img

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        if self.mode == 'val':
            return len(self.val_data)
        if self.mode=='test':
            return len(self.test_data)


def image_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], mode='train'):
    if mode == 'train':
        img_size = 224
        horizontal_flip = 0.5
        vertical_flip = 0.5

        t = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(horizontal_flip),
            transforms.RandomVerticalFlip(vertical_flip),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

    else:
        img_size = 224

        t = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

    return transforms.Compose([v for v in t])


if __name__ == '__main__':
    Doodle()
