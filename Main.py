from data import covidDataset
from train import train_val, train_val_with_LDAM
from model import myModel, se_resnet18
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
# 用来规定系统随机种子的函数，使用时直接复制即可， 便于结果复现
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#################################################################
# 执行函数以固定种子
seed_everything(0)
###############################################

class FocalLoss(nn.Module):
    def __init__(self, gamma=3, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        '''
        :param cls_num_list : 每个类别样本个数
        :param max_m : LDAM中最大margin参数,default =0.5
        :param weight :
        :param s : 缩放因子,控制logits的范围
        '''
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))  # n_j 四次方根的倒数
        m_list = m_list * (max_m / np.max(m_list))  # 归一化，C相当于 max_m/ np.max(m_list)，确保没有大于max_m的

        m_list = torch.FloatTensor(m_list).cuda()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        x = x.cuda()
        target = target.cuda()
        index = torch.zeros_like(x, dtype=torch.uint8).cuda()  # 创建一个跟X一样的tensor
        index.scatter_(1, target.data.view(-1, 1), 1)  # 将每一行对应的target的序号设为1，其余保持为0
        index_float = index.type(torch.FloatTensor).cuda()

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # 矩阵乘法，不同类别有不同的margin
        batch_m = batch_m.view((-1, 1))  # 变形之后，每一行与一个margin相减
        x_m = x - batch_m

        output = torch.where(index, x_m, x)  # 只有GT类会与margin相减
        return F.cross_entropy(self.s * output, target,
                               weight=self.weight)  # 通过一个缩放因子来放大logits，从而在使用softmax函数时增加计算结果的稳定性


# 主函数部分
# 使用字典定义训练和预测所需的配置参数
config = {
    "device": "cuda",  # 计算设备
    "epochs": 50,  # 训练的总轮数
    "batch_size": 64,  # 每个批次的数据量
    "num_classes": 2,
    "lr": 0.001,
    "path": r"D:\code\LKM\CovidImageClassification\CovidXRay",  # 训练数据文件路径, r表示去除转义字符
    "save_path": "model_save/model.pth"  # 模型保存路径
}


# 实例化模型,Dataset,Dataloader,使用优化器AdamW
train_dataset = covidDataset(config["path"], "train")
train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True)
val_dataset = covidDataset(config["path"], "val")
val_loader = DataLoader(val_dataset, config["batch_size"], shuffle=True)
loss = FocalLoss()
# loss = LDAMLoss
# loss = nn.CrossEntropyLoss()

model = myModel(num_classes=2, model_name='ResNet18')
optimizer = torch.optim.AdamW(model.parameters(), config["lr"], weight_decay=1e-2)
start_time = time.time()
train_val(model, train_loader, val_loader, optimizer, loss, config["epochs"], config["device"], config["save_path"])
# train_val_with_LDAM(model, train_loader, val_loader, optimizer, loss, config["epochs"], config["device"], config["save_path"])
end_time = time.time()
print("%s训练完成，用时：%.2f 秒" % (model.get_name(), (end_time - start_time)))

model = se_resnet18(pretrained=True, num_classes=2)
optimizer = torch.optim.AdamW(model.parameters(), config["lr"], weight_decay=1e-2)
start_time = time.time()
train_val(model, train_loader, val_loader, optimizer, loss, config["epochs"], config["device"], config["save_path"])
end_time = time.time()
print("%s训练完成，用时：%.2f 秒" % (model.get_name(), (end_time - start_time)))


