import numpy as np


class Evaluate(object):
    def __init__(self, numClass=3):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixel_Accuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def compute_mIou(self):
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)


if __name__ == '__main__':
    imgPredict = np.array([[[0, 0, 1],
                           [1, 2, 1]],
                           [[0, 0, 1],
                            [1, 2, 1]]
                           ])
    imgLabel = np.array([[[0, 0, 1],
                         [1, 2, 1]],
                         [[0, 0, 1],
                          [1, 2, 1]]
                         ])
    print(imgLabel.shape)
    metric = Evaluate(3)  # 3表示有3个分类
    metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixel_Accuracy()
    mIoU = metric.compute_mIou()
    print('pa is : %f' % pa)
    print('mIoU is : %f' % mIoU)