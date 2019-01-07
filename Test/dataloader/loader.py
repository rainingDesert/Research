import pandas as pd
import numpy as np
import random
from torchvision import transforms
from PIL import Image

'''
使用二分类观察同一object与不同object进行分类的CAM
'''


class Binary_dataset:
    def __init__(self, args, query_cls, data_csv_path='data.csv', mode='train', img_size=224):
        self.data = pd.read_csv(args.csv_path)
        self.cls = set(self.data['cls'])
        self.mode = mode
        self.args = args
        self.img_size = img_size

        self.query_data = self.data[self.data['cls'] == query_cls]
        self.query_data['label'] = 1

        self.cls.remove(query_cls)
        self.data.drop(self.query_data.index, inplace=True)
        random_cls = random.sample(self.cls, 1)[0]
        self.random_data = self.data[self.data['cls'] == random_cls]
        self.random_data['label'] = 0

        self.dataset = pd.concat([self.query_data, self.random_data])

        self.train_dataset = self.dataset[self.dataset['is_val'] == 0]
        self.val_dataset = self.dataset[self.dataset['is_val'] == 1]

        self.train_dataset.reset_index(inplace=True, drop=True)
        self.val_dataset.reset_index(inplace=True, drop=True)

        if self.mode == 'train':
            self.cur_data = self.train_dataset
        elif self.mode == 'val':
            self.cur_data = self.val_dataset

    def __getitem__(self, index):
        item = self.cur_data.loc[index]

        img_id = item['ImageId']
        path = item['path']
        label = item['label']
        bbox = item['bbox']

        raw_img = Image.open(self.args.data_root_path + path).convert('RGB')
        img = image_transform(img_size=self.img_size, mode=self.mode)(raw_img)

        return img_id, img, label, bbox

    def to_train(self):
        self.mode = 'train'
        self.cur_data = self.train_dataset

    def to_val(self):
        self.mode = 'val'
        self.cur_data = self.val_dataset

    def __len__(self):
        return len(self.cur_data)


def image_transform(img_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], mode='train'):
    if mode == 'train':
        horizontal_flip = 0.5
        vertical_flip = 0.5

        t = [
            transforms.RandomResizedCrop(size=img_size),
            transforms.RandomHorizontalFlip(horizontal_flip),
            transforms.RandomVerticalFlip(vertical_flip),
            transforms.ColorJitter(saturation=0.4, brightness=0.4, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

    else:
        t = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

    return transforms.Compose([v for v in t])
