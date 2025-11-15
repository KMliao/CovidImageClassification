import torch
import torch.nn as nn
from torchvision import models


class myModel(nn.Module):
    def __init__(self, num_classes, model_name='resnet18'):
        super(myModel, self).__init__()
        self.name = model_name
        if model_name == 'ResNet18':
            model = models.resnet18(pretrained=True)  # 从网络下载模型  pretrain true 使用参数和架构， false 仅使用架构。
            model.to("cuda")
            num_ftrs = model.fc.in_features  # 分类头的输入维度
            model.fc = nn.Linear(num_ftrs, num_classes)  # 删掉原来分类头， 更改最后一层为想要的分类数的分类头。
            self.model = model
        elif model_name == 'VGG':
            model = models.vgg19(pretrained=True)
            model.to("cuda")
            num_ftrs = model.classifier[6].in_features  # 获取VGG的分类头
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)  # 删掉原来分类头， 更改最后一层为想要的分类数的分类头。
            self.model = model
        elif model_name == 'AlexNet':
            model = models.alexnet(pretrained=True)
            model.to("cuda")
            num_ftrs = model.classifier[6].in_features  # 获取AlexNet的分类头
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)  # 删掉原来分类头， 更改最后一层为想要的分类数的分类头。
            self.model = model
        elif model_name == 'DenseNet':
            model = models.densenet121(pretrained=True)
            model.to("cuda")
            num_ftrs = model.classifier.in_features  # 获取DenseNet的分类头的输入维度
            model.classifier = nn.Linear(num_ftrs, num_classes)  # 删掉原来分类头， 更改最后一层为想要的分类数的分类头。
            self.model = model
        elif model_name == 'MobileNets':
            model = models.mobilenet_v3_large(pretrained=True)
            # 获取原始分类头的输入特征数
            num_ftrs = model.classifier[3].in_features  # MobileNetV3的分类头是Sequential，最后一层是Linear
            model.classifier[3] = nn.Linear(num_ftrs, num_classes)  # 删掉原来分类头， 更改最后一层为想要的分类数的分类头。
            self.model = model

    def forward(self, x):
        return self.model(x)

    def get_name(self):
        return self.name


# 以下是SE改进部分
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1  # BasicBlock 的扩张系数是 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes * self.expansion, reduction)  # SE 模块作用在输出通道上
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # 在残差连接前应用 SE 模块

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(SEResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-18 的 4 个层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_name(self):
        return 'SE_ResNet18'


# 创建 SE-ResNet-18
def se_resnet18(pretrained=True, num_classes=1000):
    model = SEResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        # 加载预训练的 ResNet-18 权重
        pretrained_dict = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet18-5c106cde.pth')
        model_dict = model.state_dict()

        # 过滤掉不匹配的权重（主要是 fc 层）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


if __name__ == "__main__":
    # 实例化模型，启用预训练
    model = myModel(num_classes=12, pretrained=True)
    # 输入示例
    input_tensor = torch.randn(32, 3, 224, 224)
    output = model(input_tensor)
    print("Output shape:", output.shape)
