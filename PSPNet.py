import torch
from torchvision import models
import cv2
import numpy as np
from torch import nn
import torch.nn.functional as F
from Iou import Evaluate


#提取特征部分, resnet50的前四层
class Resnet_psp(nn.Module):
    def __init__(self, classes, asu=True, dropout=0.5):
        super(Resnet_psp, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        # 修改第三层和第四层的卷积层
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        self.layer4 = self.resnet.layer4
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        # 引出第三层计算辅助损失
        self.asu = asu
        self.asu_layer = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x_4 = self.layer4(x_3)    # 特征图，[b, 2048, h/8, w/8]
        if self.asu:
            x_3 = self.asu_layer(x_3)   # 用于计算辅助损失，从layer3引出
        return x_3, x_4


# 池化部分
class PPM(nn.Module):
    def __init__(self, in_ch, out_ch, out_scale):
        super(PPM, self).__init__()
        self.PSP = []      # 四种不同的池化层
        # 池化加卷积
        for size_out in out_scale:
            self.PSP.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size_out),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
        self.PSP = nn.ModuleList(self.PSP)

    def forward(self, x):
        w, h = x.size(2), x.size(3)
        out = [x]
        for psp in self.PSP:
            out.append(F.interpolate(psp(x), size=(w, h), mode='bilinear', align_corners=True))  # 上采样到同样大小
        return torch.cat(out, 1)  # cat到一起


# 整体网络结构
class PSPNet(nn.Module):
    def __init__(self, classes=2, in_ch=2048, out_ch=512):
        super(PSPNet, self).__init__()
        self.out_scale = [1, 2, 3, 6]
        self.res_psp = Resnet_psp(classes=classes)
        self.ppm_psp = PPM(in_ch=in_ch, out_ch=out_ch, out_scale=self.out_scale)

        # 预测结果
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch*2, classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(classes),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )

    def forward(self, x):
        w, h = x.size(2), x.size(3)
        x3, x4 = self.res_psp(x)
        out = self.ppm_psp(x4)
        out = self.layer(out)
        out = F.interpolate(out, size=(w, h), mode='bilinear', align_corners=True)
        out_x3 = F.interpolate(x3, size=(w, h), mode='bilinear', align_corners=True)

        return out, out_x3  # 返回第三层和最终的预测结果


# 计算辅助损失，主干损失
class LossOfaux(nn.Module):
    def __init__(self):
        super(LossOfaux, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, out, out_3, y=None):
        main_loss = self.loss(out, y)
        aux_loss = self.loss(out_3, y)

        return main_loss + 0.4 * aux_loss


# 计算Dice损失
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, x, y):
        x1 = x.view(-1)
        y1 = y.view(-1)
        intersection = (x1 * y1).sum()
        dice_loss = 1 - 2 * intersection / (x1.sum() + y1.sum() + self.smooth)
        return dice_loss


if __name__ == '__main__':
    input = torch.randn((2, 3, 473, 473))
    label = torch.randn((2, 473, 473))
    model = PSPNet()
    out_merge, out_x3 = model(input)
    prob_out, prob_out3 = torch.softmax(out_merge, 1), torch.softmax(out_x3, 1)
    pre = torch.argmax(prob_out, 1)
    Loss = LossOfaux()
    Loss_dice = DiceLoss()
    loss = Loss_dice(pre, label)
    print('loss', loss)
    loss.requires_grad_()
    loss.backward()
    print(out_merge.size())
    print(pre.size())

