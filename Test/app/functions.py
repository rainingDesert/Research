import numpy as np
import torch
import shutil
import torch.nn.functional as F
import matplotlib.pyplot as plt
import PIL


def cal_acc(logits, label):
    pre = torch.argmax(logits, dim=-1)

    return torch.mean((pre == label).type(torch.FloatTensor)).item()


def save_imgs_cams_by_id(args, img_id, logits, cams, dataset):
    data = dataset.cur_data


    for index, id in enumerate(img_id):
        path = data[data['ImageId'] == id]['path'].values[0]

        shutil.copyfile(args.data_root_path + path, '../save/train/imgs/{}.JPEG'.format(id))

        target_cam = cams[index].unsqueeze(0)
        raw_img = PIL.Image.open(args.data_root_path + path).convert('RGB')
        up_target_cam = F.upsample(target_cam, size=(raw_img.size[1], raw_img.size[0]), mode='bilinear',
                                   align_corners=True).squeeze()
        cam = up_target_cam[torch.argmax(logits[index], dim=-1)]
        plt.imshow(cam.cpu().data.numpy())
        plt.savefig('../save/train/imgs/cam_{}.JPEG'.format(id))
