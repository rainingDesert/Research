from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from xml.dom import minidom
from torchvision import transforms

img_dir = '../../../../../Data/VOCdevkit/VOC2012/JPEGImages/'
class_label_dir = '../../../../../Data/VOCdevkit/VOC2012/ImageSets/Main/'
anotation_dir = '../../../../../Data/VOCdevkit/VOC2012/Annotations/'
seg_class_dir = '../../../../../Data/VOCdevkit/VOC2012/SegmentationClass/'
seg_obj_dir = '../../../../../Data/VOCdevkit/VOC2012/SegmentationObject/'
cotegory = ['dog', 'sofa', 'bicycle', 'aeroplane', 'pottedplant', 'car', 'horse', 'bird', 'bottle', 'cow', 'cat',
            'tvmonitor', 'person', 'train', 'boat', 'diningtable', 'sheep', 'chair', 'motorbike', 'bus']


# mode=['seg_class','seg_obj]
class VOC(Dataset):
    def __init__(self, mode='train', train_val_split=0.9, img_transform=None):
        self.mode = mode
        self.data = pd.read_csv('../class_only_data.csv')
        self.test_seg_class_data = pd.read_csv('../class_seg_data.csv')
        self.test_seg_obj_data = pd.read_csv('../obj_seg_data.csv')

        self.train_val_split = train_val_split

        train_num = int(len(self.data) * self.train_val_split)
        self.train_data = self.data.loc[self.data.index[:train_num]]
        self.val_data = self.data.loc[self.data.index[train_num:]]

        self.img_transform = img_transform

    def __getitem__(self, index):
        if self.mode == 'train':
            item = self.train_data.loc[index]
            img_name = item['img_name']
            class_label = item['img_class']

            img_path = img_dir + img_name + '.jpg'
            img = Image.open(img_path).convert('RGB')

            if self.img_transform:
                img = self.img_transform(img)

            return img, class_label
        elif self.mode == 'val':
            item = self.val_data.loc[index]
            img_name = item['img_name']
            class_label = item['img_class']

            img_path = img_dir + img_name + '.jpg'
            img = Image.open(img_path).convert('RGB')

            if self.img_transform:
                img = self.img_transform(img)

            return img, class_label

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)


def image_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    configure_doc = minidom.parse('../configure.xml')
    root = configure_doc.documentElement

    preprocess = root.getElementsByTagName('preprocess')[0]
    img_size = int(preprocess.getElementsByTagName('img_size')[0].firstChild.data)
    horizontal_flip = float(preprocess.getElementsByTagName('horizontal_flip')[0].firstChild.data)

    t = [
        transforms.Resize((img_size, img_size)) if img_size is not None else None,
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        #transforms.RandomVerticalFlip(vertical_flip) if vertical_flip is not None else None,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    return transforms.Compose([v for v in t if v is not None])


if __name__ == '__main__':
    dataset = VOC(img_transform=image_transform())
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for step, (b_x, b_y) in enumerate(dataloader):
        print()
