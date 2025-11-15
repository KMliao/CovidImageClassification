import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms, autoaugment
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 临时解决 OpenMP 冲突

HW = 224  # 定义全局变量规定图像分类任务中图像的的标准大小
test_transform = transforms.Compose([
    transforms.ToTensor(),
])              # 测试集只需要转为张量

# 自定义高斯噪声类
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# 改进后的训练数据增强
train_transform = transforms.Compose([
    transforms.ToPILImage(),                          # 转换为PIL图像
    transforms.RandomResizedCrop(HW, scale=(0.9, 1.1)), # 随机裁剪并缩放
    transforms.RandomHorizontalFlip(),                # 随机水平翻转
    transforms.RandomRotation(10),                    # 随机旋转±10度
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # 随机调整亮度和对比度
    transforms.GaussianBlur(kernel_size=3),           # 添加高斯模糊
    transforms.ToTensor(),                            # 转换为张量
    AddGaussianNoise(0., 0.01)                       # 添加高斯噪声
])


def readFile(path):
    x = []
    y = []
    class_counts = [0, 0]  # 有2个类别：NORMAL 和 PNEUMONIA
    for i in range(2):
        if i == 0:
            imgDirPath = path + '/NORMAL/'
        else:
            imgDirPath = path + '/PNEUMONIA/'
        imgList = os.listdir(imgDirPath)
        class_counts[i] = len(imgList)  # 记录每个类别的样本数
        xi = np.zeros((len(imgList), HW, HW, 3), dtype=np.uint8)
        # yi = np.zeros((len(imgList)), dtype=np.uint8)
        yi = np.zeros((len(imgList)), dtype=np.int64)
        for j, each in enumerate(imgList):
            imgPath = imgDirPath + each
            img = Image.open(imgPath).convert('RGB')
            img = img.resize((HW, HW))
            xi[j, ...] = img
            yi[j] = i
        if i == 0:
            x = xi
            y = yi
        else:
            x = np.concatenate((x, xi), axis=0)
            y = np.concatenate((y, yi), axis=0)
    print('读入图像%d个 ' % len(x))
    return x, y, class_counts

class covidDataset(Dataset):
    def __init__(self, path, mode):
        pathDict = {'train': 'train', 'val': 'test'}
        imgPaths = path + '/' + pathDict[mode]  # 定义路径
        self.mode = mode
        self.transform = None  # 提前定义图片的增广函数
        self.x = None
        self.y = None
        if mode == "train":
            self.x, self.y, self.class_counts = readFile(imgPaths)
            self.transform = train_transform
        elif mode == "val":
            self.x, self.y, self.class_counts = readFile(imgPaths)
            self.transform = test_transform

    def __getitem__(self, index):
        orix = self.x[index]  # 先根据下标读出原始图片
        if self.transform is not None:  # 若图片需要进行增广
            xT = self.transform(orix)  # 把原始图片进行增广
        else:
            xT = torch.tensor(orix).float()

        if self.y is not None:
            return xT, self.y[index], orix  # 分别返回增广后的图片，标签，原始图片
        else:
            return xT, orix

    def __len__(self):
        return len(self.x)

    def getClassCounts(self):
        return self.class_counts
