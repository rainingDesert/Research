from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import PIL
from xml.dom import minidom
from torchvision import transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


class Doodle(Dataset):
    def __init__(self, train_csv_path, val_csv_path, test_csv_path, args, mode='train', val_frac=0.00003):
        self.args = args
        self.mode = mode
        self.train_data = pd.read_csv(train_csv_path)
        self.val_data = pd.read_csv(val_csv_path)
        self.test_data = pd.read_csv(test_csv_path)

    def __getitem__(self, index):
        if self.mode == 'train':
            item = self.train_data.loc[index]

            img_path = item['Path']
            img_name = img_path.split('/')[-1]
            img_label = torch.squeeze(torch.from_numpy(np.array(int(str(item['Label']).strip())).reshape(-1)))

            img_files = os.listdir(img_path)
            sliced_imgs = []
            for img_file in img_files:
                if '_' in img_file:
                    img = Image.open(img_path + '/' + img_file)
                    img = image_transform(mode='train')(img)
                    img = img.unsqueeze(0)
                    sliced_imgs.append(img)

            img = torch.cat(sliced_imgs)

            return img_name, img, img_label

        elif self.mode == 'val':
            item = self.val_data.loc[index]

            img_path = item['Path']
            img_name = img_path.split('/')[-1]
            img_label = torch.squeeze(torch.from_numpy(np.array(int(str(item['Label']).strip())).reshape(-1)))

            img_files = os.listdir(img_path)
            sliced_imgs = []
            for img_file in img_files:
                if '_' in img_file:
                    img = Image.open(img_path + '/' + img_file)
                    img = image_transform(mode='train')(img)
                    img = img.unsqueeze(0)
                    sliced_imgs.append(img)

            img = torch.cat(sliced_imgs)

            return img_name, img, img_label

        else:
            item = self.test_data.loc[index]

            img_path = item['path']
            img_name = img_path.split('/')[-1]
            img_path='/home/kzy/Data/Doodle/test/images/'+img_name

            img_files = os.listdir(img_path)
            sliced_imgs = []
            for img_file in img_files:
                if '_' in img_file:
                    img = Image.open(img_path + '/' + img_file)
                    img = image_transform(mode='test')(img)
                    img = img.unsqueeze(0)
                    sliced_imgs.append(img)

            img = torch.cat(sliced_imgs)

            return img_name, img
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        if self.mode == 'val':
            return len(self.val_data)
        if self.mode == 'test':
            return len(self.test_data)


def image_transform(mode='train'):
    img_size = 28

    if mode == 'train':
        horizontal_flip = 0.5
        vertical_flip = 0.5

        t = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(horizontal_flip),
            transforms.RandomVerticalFlip(vertical_flip),
            transforms.ToTensor(),
        ]

    else:
        t = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]

    return transforms.Compose([v for v in t])


if __name__ == '__main__':
    Doodle()
