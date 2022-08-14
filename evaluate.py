from PSPNet import PSPNet
import torch
from torch.utils.data import DataLoader
from dataset import dataset
from Iou import Evaluate


def resultTest(img_path, label_path, model_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('{} is uesd!'.format(device))
    test_datas = dataset(img_path=img_path, label_path=label_path, file_path='images/test.txt')
    test_loader = DataLoader(test_datas, batch_size=batch_size, shuffle=False, drop_last=True)
    print('test:', len(test_datas))
    model = PSPNet()

    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()
    metric = Evaluate(2)
    test_pa = 0.0
    test_miou = 0.0
    count = 0
    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            output, out3 = model(img)  # out3是resnetlayer3引出的分支
            prob, prob3 = torch.softmax(output, 1), torch.softmax(out3, 1)
            pre = torch.argmax(prob, 1)
            metric.addBatch(pre, label)
            pa = metric.pixel_Accuracy()
            miou = metric.compute_mIou()
            test_pa += pa
            test_miou += miou
            count += 1
    test_pa, test_miou = test_pa / count, test_miou / count
    print('-'*100)
    print('test: pixel accuracy:', test_pa)
    print('test: miou:', test_miou)


if __name__ == '__main__':
    img_path = 'images/horse'
    label_path = 'images/mask'
    model_path = 'model_path/model_51miou_0.745.pth'
    batch_size = 2
    resultTest(img_path, label_path, model_path, batch_size)