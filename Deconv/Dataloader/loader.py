import pandas as pd
from PIL import Image
from torchvision import transforms


class CUB_Loader:
    def __init__(self, args, mode='train'):
        self.data_csv = pd.read_csv(args.csv_path)
        self.data_csv = self.data_csv[self.data_csv['cls'] == args.cls]
        self.data_csv.loc[:, 'cls'] = 0
        self.mode = mode
        self.args = args
        self.img_size = args.train_img_size
        self.class_nums = args.class_nums

        self.train_csv = self.data_csv[self.data_csv['is_train'] == 1]
        self.val_csv = self.data_csv[self.data_csv['is_train'] == 0]
        self.train_csv.reset_index(drop=True, inplace=True)
        self.val_csv.reset_index(drop=True, inplace=True)

        if self.mode == 'train':
            self.cur_csv = self.train_csv
        else:
            self.cur_csv = self.val_csv

    def __getitem__(self, index):
        item = self.cur_csv.loc[index]

        img_id = item['id']
        path = item['path']
        label = item['cls']
        bbox = item['bbox']

        raw_img = Image.open(self.args.data_root_path + path).convert('RGB')
        img = self.image_transform(img_size=self.img_size, mode=self.mode)(raw_img)

        return img_id, img, label, bbox

    def to_train(self):
        self.mode = 'train'
        self.cur_csv = self.train_csv

    def to_val(self):
        self.mode = 'val'
        self.cur_csv = self.val_csv

    def __len__(self):
        return len(self.cur_csv)

    @staticmethod
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
