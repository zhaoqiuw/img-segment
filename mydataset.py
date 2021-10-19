import os
import numpy as np
import cv2
from pre_process import data_augment
from config import cfg


# 自定义数据集
class DatasetGenerator:
    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_path = os.listdir(os.path.join(self.root_dir, self.img_dir))
        self.label_path = [
            x.strip('.png') + '_mask.png' for x in self.img_path
        ]

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.img_dir, img_name)
        img = cv2.imread(img_item_path, cv2.IMREAD_COLOR)
        label_name = self.label_path[index]
        label_item_path = os.path.join(self.root_dir, self.label_dir,
                                       label_name)
        label = cv2.imread(label_item_path, cv2.IMREAD_GRAYSCALE)
        data_augment(img, label)
        img = np.transpose(img, (2, 0, 1))
        label = np.reshape(label, (1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
        img = img / 255.
        label = label / 255. 
        return img, label

    def __len__(self):
        return len(self.img_path)
