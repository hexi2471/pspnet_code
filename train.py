import torch
from dataset import dataset
from PSPNet import PSPNet, LossOfaux, DiceLoss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from Iou import Evaluate
import matplotlib.pyplot as plt


# 训练部分
def train(img_path, label_path, epochs, learn_rate, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('{} is uesd!'.format(device))
    datas = dataset(img_path, label_path)
    print('train images:{}'.format(len(datas)))
    train_loader = DataLoader(datas, batch_size=batch_size, shuffle=True, drop_last=True)

    model = PSPNet()  # PSPNet模型
    Loss = LossOfaux()  # 损失函数，主干损失+辅助损失
    Loss_dice = DiceLoss()  # Dice损失
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, betas=(0.99, 0.099), eps=1e-7)  # Adam优化器
    schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # 自动调整学习率

    metric = Evaluate(2)

    model.to(device)
    Loss.to(device)

    train_loss = []
    train_mIou = []
    train_pa = []
    train_epoch = []
    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        t_pa = 0.0
        t_miou = 0.0
        count = 0
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            output, out3 = model(img)  # out3是resnetlayer3引出的分支
            prob, prob3 = torch.softmax(output, 1), torch.softmax(out3, 1)
            pre = torch.argmax(prob, 1)
            loss = Loss(prob, prob3, label.long()) + Loss_dice(pre, label)  # 计算损失
            t_loss += loss.item()
            metric.addBatch(pre, label)
            # 计算pixel accuracy 和 mIoU
            pa = metric.pixel_Accuracy()
            miou = metric.compute_mIou()
            t_pa += pa
            t_miou += miou

            optimizer.zero_grad()
            # loss.requires_grad_()
            loss.backward()
            optimizer.step()
            count += 1
        schedule.step()

        t_loss, t_pa, t_miou = t_loss / count, t_pa / count, t_miou / count
        print("=" * 100)
        print('{}/{}: pixel accuracy:{}  mIOU:{}    loss:{}'.format(epoch + 1, epochs, t_pa, t_miou, t_loss))
        # 每5轮保存一次训练结果并进行验证
        if epoch % 10 == 0:
            train_epoch.append(str(epoch + 1))
            train_pa.append(t_pa)
            train_mIou.append(t_miou)
            train_loss.append(t_loss)
            save_path = 'model_path/model_' + str(epoch + 1) + 'miou_' + str(np.round(t_miou, 3)) + '.pth'
            torch.save(model.state_dict(), save_path)  # 保存模型权重
            print('model saved!')
    # 可视化结果
    plt.figure(figsize=(15, 15))
    plt.subplot(221)
    plt.ylim(0, 1)
    plt.plot(train_epoch, train_pa, label='train pixel accuracy', marker='o', markersize=5)
    # plt.plot(train_epoch, train_loss, label='train_loss', marker='o', markersize=5)
    plt.plot(train_epoch, train_mIou, label='train_mIoU', marker='x', markersize=5)
    plt.xlabel('epoch')
    plt.title('Pixel Accuracy and mIOU')
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.subplot(222)
    # plt.ylim(0, 1)
    # plt.plot(train_epoch, train_pa, label='train pixel accuracy', marker='o', markersize=5)
    plt.plot(train_epoch, train_loss, label='train_loss', marker='o', markersize=5)
    # plt.plot(train_epoch, train_mIou, label='train_mIoU', marker='x', markersize=5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    img_path = 'images/horse'
    label_path = 'images/mask'
    epochs = 10
    ratio = 0.85
    learn_rate = 1e-3
    batch_size = 2
    train(img_path, label_path, epochs, ratio, learn_rate, batch_size)