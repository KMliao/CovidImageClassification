import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import f1_score  # 导入f1_score函数
# 解决matplotlib报错
import matplotlib
matplotlib.use('TkAgg')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 临时解决 OpenMP 冲突

# 定义训练和验证过程，包括参数更新和损失可视化
def train_val(model, trainloader, valloader, optimizer, loss, epochs, device, save_):
    model = model.to(device)  # 将模型移动到指定设备（CPU或GPU）
    plt_train_loss = []  # 存储每个epoch的训练损失
    plt_val_loss = []    # 存储每个epoch的验证损失
    plt_train_acc = []   # 存储每个epoch的训练准确率
    plt_val_acc = []     # 存储每个epoch的验证准确率
    plt_val_f1 = []      # 新增：存储每个epoch的验证F1分数
    max_acc = 0.0        # 记录最佳验证准确率，用于保存模型

    for epoch in range(epochs):
        start_time = time.time()  # 记录当前epoch开始时间
        # 训练阶段
        model.train()  # 设置模型为训练模式（启用Dropout、BatchNorm等）
        train_loss = 0.0  # 累加训练损失
        train_acc = 0.0   # 累加训练正确预测数
        for data in tqdm(trainloader):  # 使用tqdm显示训练进度
            optimizer.zero_grad()  # 清空优化器的梯度
            x, target = data[0].to(device), data[1].to(device)  # 输入数据和标签移到设备
            pred = model(x)  # 前向传播，得到预测结果
            train_bat_loss = loss(pred, target)  # 计算当前批次的损失
            train_bat_loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
            train_loss += train_bat_loss.detach().cpu().item()  # 累加损失（移到CPU并转为标量）
            # 计算当前批次的正确预测数
            train_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == target.cpu().numpy())
        plt_train_loss.append(train_loss / trainloader.dataset.__len__())  # 计算平均训练损失
        plt_train_acc.append(train_acc / trainloader.dataset.__len__())   # 计算平均训练准确率

        # 验证阶段
        model.eval()  # 设置模型为评估模式（关闭Dropout、BatchNorm更新）
        val_loss = 0.0  # 累加验证损失
        val_acc = 0.0   # 累加验证正确预测数
        all_val_preds = []  # 新增：存储所有验证预测标签
        all_val_targets = []  # 新增：存储所有验证真实标签
        with torch.no_grad():  # 禁用梯度计算以节省内存和加速
            for data in valloader:  # 遍历验证数据加载器
                val_x, val_target = data[0].to(device), data[1].to(device)  # 输入和标签移到设备
                val_pred = model(val_x)  # 前向传播，得到验证预测
                val_bat_loss = loss(val_pred, val_target)  # 计算当前批次验证损失
                val_loss += val_bat_loss.detach().cpu().item()  # 累加验证损失
                # 计算当前批次的正确预测数
                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == val_target.cpu().numpy())
                # 新增：收集预测和真实标签用于F1分数计算
                val_pred_labels = np.argmax(val_pred.cpu().numpy(), axis=1)  # 获取预测类别
                val_target_labels = val_target.cpu().numpy()  # 获取真实类别
                all_val_preds.extend(val_pred_labels)  # 将当前批次预测添加到列表
                all_val_targets.extend(val_target_labels)  # 将当前批次真实标签添加到列表

        # 保存最佳模型（基于验证准确率）
        if val_acc > max_acc:
            torch.save(model, save_)  # 保存模型到指定路径
            max_acc = val_acc  # 更新最大准确率

        # 计算并存储平均验证损失和准确率
        plt_val_loss.append(val_loss / valloader.dataset.__len__())  # 平均验证损失
        plt_val_acc.append(val_acc / valloader.dataset.__len__())    # 平均验证准确率

        # 新增：计算F1分数
        val_f1 = f1_score(all_val_targets, all_val_preds, average='macro')  # 计算F1分数（macro平均）
        plt_val_f1.append(val_f1)  # 存储当前epoch的F1分数

        # 打印当前epoch的训练和验证结果
        print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f | ValAcc: %3.6f ValLoss: %3.6f | ValF1: %3.6f' % \
              (epoch + 1, epochs, time.time() - start_time, plt_train_acc[-1], plt_train_loss[-1],
               plt_val_acc[-1], plt_val_loss[-1], plt_val_f1[-1]))
    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title('loss')
    plt.legend(['train', 'val'])
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'val'])
    plt.show()

    plt.plot(plt_val_f1)
    plt.title('F1 Scores')
    plt.legend(['val'])
    plt.savefig('F1分数-%s.png' % model.get_name())
    plt.show()


def train_val_with_LDAM(model, trainloader, valloader, optimizer, loss, epochs, device, save_):
    model = model.to(device)  # 将模型移动到指定设备（CPU或GPU）
    plt_train_loss = []  # 存储每个epoch的训练损失
    plt_val_loss = []    # 存储每个epoch的验证损失
    plt_train_acc = []   # 存储每个epoch的训练准确率
    plt_val_acc = []     # 存储每个epoch的验证准确率
    plt_val_f1 = []      # 新增：存储每个epoch的验证F1分数
    max_acc = 0.0        # 记录最佳验证准确率，用于保存模型
    train_class_counts = trainloader.dataset.getClassCounts()
    val_class_counts = valloader.dataset.getClassCounts()
    loss_train = loss(train_class_counts)
    loss_val = loss(val_class_counts)

    for epoch in range(epochs):
        start_time = time.time()  # 记录当前epoch开始时间
        # 训练阶段
        model.train()  # 设置模型为训练模式（启用Dropout、BatchNorm等）
        train_loss = 0.0  # 累加训练损失
        train_acc = 0.0   # 累加训练正确预测数
        for data in tqdm(trainloader):  # 使用tqdm显示训练进度
            optimizer.zero_grad()  # 清空优化器的梯度
            x, target = data[0].to(device), data[1].to(device)  # 输入数据和标签移到设备
            pred = model(x)  # 前向传播，得到预测结果
            train_bat_loss = loss_train(pred, target)  # 计算当前批次的损失
            train_bat_loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
            train_loss += train_bat_loss.detach().cpu().item()  # 累加损失（移到CPU并转为标量）
            # 计算当前批次的正确预测数
            train_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == target.cpu().numpy())
        plt_train_loss.append(train_loss / trainloader.dataset.__len__())  # 计算平均训练损失
        plt_train_acc.append(train_acc / trainloader.dataset.__len__())   # 计算平均训练准确率

        # 验证阶段
        model.eval()  # 设置模型为评估模式（关闭Dropout、BatchNorm更新）
        val_loss = 0.0  # 累加验证损失
        val_acc = 0.0   # 累加验证正确预测数
        all_val_preds = []  # 新增：存储所有验证预测标签
        all_val_targets = []  # 新增：存储所有验证真实标签
        with torch.no_grad():  # 禁用梯度计算以节省内存和加速
            for data in valloader:  # 遍历验证数据加载器
                val_x, val_target = data[0].to(device), data[1].to(device)  # 输入和标签移到设备
                val_pred = model(val_x)  # 前向传播，得到验证预测
                val_bat_loss = loss_val(val_pred, val_target)  # 计算当前批次验证损失
                val_loss += val_bat_loss.detach().cpu().item()  # 累加验证损失
                # 计算当前批次的正确预测数
                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == val_target.cpu().numpy())
                # 新增：收集预测和真实标签用于F1分数计算
                val_pred_labels = np.argmax(val_pred.cpu().numpy(), axis=1)  # 获取预测类别
                val_target_labels = val_target.cpu().numpy()  # 获取真实类别
                all_val_preds.extend(val_pred_labels)  # 将当前批次预测添加到列表
                all_val_targets.extend(val_target_labels)  # 将当前批次真实标签添加到列表

        # 保存最佳模型（基于验证准确率）
        if val_acc > max_acc:
            torch.save(model, save_)  # 保存模型到指定路径
            max_acc = val_acc  # 更新最大准确率

        # 计算并存储平均验证损失和准确率
        plt_val_loss.append(val_loss / valloader.dataset.__len__())  # 平均验证损失
        plt_val_acc.append(val_acc / valloader.dataset.__len__())    # 平均验证准确率

        # 新增：计算F1分数
        val_f1 = f1_score(all_val_targets, all_val_preds, average='macro')  # 计算F1分数（macro平均）
        plt_val_f1.append(val_f1)  # 存储当前epoch的F1分数

        # 打印当前epoch的训练和验证结果
        print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f | ValAcc: %3.6f ValLoss: %3.6f | ValF1: %3.6f' % \
              (epoch + 1, epochs, time.time() - start_time, plt_train_acc[-1], plt_train_loss[-1],
               plt_val_acc[-1], plt_val_loss[-1], plt_val_f1[-1]))
    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title('loss')
    plt.legend(['train', 'val'])
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'val'])
    plt.show()

    plt.plot(plt_val_f1)
    plt.title('F1 Scores')
    plt.legend(['val'])
    plt.savefig('F1分数-%s.png' % model.get_name())
    plt.show()
