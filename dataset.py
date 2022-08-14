import os
import torch
from torch.utils.data import Dataset
import cv2
import random
import numpy as np

class dataset(Dataset):
    def __init__(self, img_path='images/horse', label_path='images/mask', file_path='images/train.txt', ratio=0.85, split=False):
        self.img_path = img_path
        self.label_path = label_path
        self.width, self.height = 473, 473     # 调整后的图片大小
        self.files = []
        if not os.path.exists(file_path) or split:
            self.spilt_train_test(ratio=ratio)

        with open(file_path, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                self.files.append(line.rstrip('\n'))
        fp.close()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        assert len(self.files) > 0
        img, label = self.get_IM(self.files[item])  # 读取图片和gt
        img, label = self.crop(img, label)   # 调整大小
        img, label = torch.tensor(img, dtype=torch.float32), torch.tensor(label)
        img = img.permute(2, 0, 1).contiguous()  # 调整维度
        return img, label

    # 读取图片
    def get_IM(self, file_name):
        # print(file_name)
        img_p = os.path.join(self.img_path, file_name)           # 图片路径
        label_p = os.path.join(self.label_path, file_name)       # 对应gt保存路径
        img, label = cv2.imread(img_p, cv2.IMREAD_COLOR), cv2.imread(label_p, cv2.IMREAD_GRAYSCALE)
        return img / 255, label  # /255 归一化

    # 随机截取 473x473大小的图片
    def crop(self, img, label):
        w, h = img.shape[0], img.shape[1]
        # 图片大小统一成473x473
        if w != self.width or h != self.height:
            img_new = cv2.resize(img, (self.width, self.height))
            label_new = cv2.resize(label, (self.width, self.height))
            return img_new, label_new
        else:
            return img, label


    #划分训练集和测试集
    def spilt_train_test(self, ratio):
        train_path = 'images/train.txt'
        test_path = 'images/test.txt'
        assert os.path.exists(train_path) == os.path.exists(test_path)  # 保证同时存在或者同时不存在
        files = os.listdir(self.img_path)
        lens = len(files)
        index = [i for i in range(lens)]
        random.shuffle(index)  # 打乱
        num_train = int(lens * ratio)
        train_index, test_index = index[0:num_train], index[num_train:-1]
        with open(train_path, 'w') as fp:
            for i in range(len(train_index)):
                file_name = files[train_index[i]]
                fp.write(file_name + '\n')
        fp.close()
        with open(test_path, 'w') as fp:
            for i in range(len(test_index)):
                file_name = files[test_index[i]]
                fp.write(file_name + '\n')
        fp.close()


if __name__ == '__main__':
    # datas = dataset()
    # print(len(datas))
    x = [1, 3, 4]
    np.random.shuffle(x)
    print(x)
