from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from xml.dom import minidom
from torchvision import transforms
import numpy as np
import torch
import os


class CUB_BI(Dataset):
    def __init__(self, root_dir, bi_bird_data, bi_nature_data, mode='train', sliced_nums=20):
        self.mode = mode
        self.sliced_nums = sliced_nums
        self.root_dir = root_dir

        self.bi_bird_data = pd.read_csv(bi_bird_data)
        self.bi_nature_data = pd.read_csv(bi_nature_data)

        self.bi_bird_data['path'] = 'images/' + self.bi_bird_data['path']

        nature_img_length = len(self.bi_nature_data)

        self.used_bi_bird_data = self.bi_bird_data.sample(
            frac=float(float(nature_img_length) / len(self.bi_bird_data)) * 2)

        self.bi_csv = pd.concat([self.used_bi_bird_data, self.bi_nature_data])
        self.bi_csv.reset_index(drop=True, inplace=True)

        self.val_data = self.bi_csv.sample(frac=0.02)
        self.train_data = self.bi_csv.drop(index=self.val_data.index)

        self.train_data.reset_index(drop=True, inplace=True)
        self.val_data.reset_index(drop=True, inplace=True)

        # data_csv = []
        # for root, dirs, files in os.walk(root_dir + 'natures/'):
        #     for file in files:
        #         data_csv.append([file.split('.')[0], 'natures/' + file, 0])
        #
        # pd.DataFrame(data=data_csv, columns=['img_name', 'path', 'label']).to_csv('../Save/bi_nature_data.csv',
        #                                                                           index=False)

        # self.data = pd.read_csv('../Save/CUB2011/data.csv')
        #
        # self.train_data = self.data[self.data['is_train'] == 1]
        # self.test_data = self.data[self.data['is_train'] == 0]
        #
        # self.train_data.reset_index(drop=True, inplace=True)
        # self.test_data.reset_index(drop=True, inplace=True)
        #
        # self.val_data = self.test_data.sample(frac=val_frac)
        # self.test_data.drop(index=self.val_data.index, inplace=True)
        #
        # self.test_data.reset_index(drop=True, inplace=True)
        # self.val_data.reset_index(drop=True, inplace=True)
        #
        # self.train_data['label'] = self.train_data['label'] - 1
        # self.val_data['label'] = self.val_data['label'] - 1
        # self.test_data['label'] = self.test_data['label'] - 1
        #
        # self.train_data.to_csv('../Save/CUB2011/train_data.csv', index=False)
        # self.val_data.to_csv('../Save/CUB2011/val_data.csv', index=False)
        # self.test_data.to_csv('../Save/CUB2011/test_data.csv', index=False)

    def __getitem__(self, index):
        if self.mode == 'train':
            item = self.train_data.loc[index]

            img_name = item['img_name']
            img_path = item['path']
            img_label = torch.squeeze(torch.from_numpy(np.array(int(str(item['label']).strip())).reshape(-1)))

            raw_img = Image.open(self.root_dir + img_path).convert('RGB')

            sliced_imgs = []
            for i in range(self.sliced_nums):
                sliced_img = image_transform(mode='train')(raw_img)
                sliced_img = sliced_img.unsqueeze(0)
                sliced_imgs.append(sliced_img)

            img = torch.cat(sliced_imgs)

            return img_name, img, img_label

        elif self.mode == 'val':
            item = self.val_data.loc[index]
            img_name = item['img_name']
            img_path = item['path']
            img_label = torch.squeeze(torch.from_numpy(np.array(int(str(item['label']).strip())).reshape(-1)))

            raw_img = Image.open(self.root_dir + img_path).convert('RGB')

            sliced_imgs = []
            for i in range(self.sliced_nums):
                sliced_img = image_transform(mode='train')(raw_img)
                sliced_img = sliced_img.unsqueeze(0)
                sliced_imgs.append(sliced_img)

            img = torch.cat(sliced_imgs)

            return img_name, img, img_label

        # else:
        #     item = self.test_data.loc[index]
        #     img_name = item['img_name']
        #     img_path = item['path']
        #     img_label = torch.squeeze(torch.from_numpy(np.array(int(str(item['label']).strip())).reshape(-1)))
        #     img = Image.open(self.root_dir + 'images/' + img_path).convert('RGB')
        #     img = image_transform(mode='test')(img)
        #
        #     return img_name, img, img_label

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        if self.mode == 'val':
            return len(self.val_data)
        if self.mode == 'test':
            return len(self.test_data)


def image_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], mode='train'):
    img_size = 256
    crop_size = 64
    if mode == 'train':
        horizontal_flip = 0.5
        vertical_flip = 0.5

        t = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(horizontal_flip),
            transforms.RandomVerticalFlip(vertical_flip),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

    else:
        t = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

    return transforms.Compose([v for v in t])


if __name__ == '__main__':
    CUB(root_dir='../../../../../Data/CUB2011/CUB_200_2011/', bi_bird_data='../Save/bi_bird_data.csv',
        bi_nature_data='../Save/bi_nature_data.csv')
