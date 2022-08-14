import cv2

from PSPNet import PSPNet
import torch
import numpy as np

img_path = 'images/horse/horse126.png'
mask_path = 'images/mask/horse126.png'
model_path = 'model_path/model_51miou_0.755.pth'

model = PSPNet()
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc:storage))
model.eval()

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (473, 473))

x = torch.tensor(img / 255, dtype=torch.float32)
x = x.permute(2, 0, 1).contiguous()  # 调整维度
x = x.unsqueeze(0)

out, out3 = model(x)

prob = torch.softmax(out, 1)
pre = torch.argmax(prob, 1)
pre = pre.squeeze(0)

img_pre = pre.detach().numpy()
img_pre = img_pre.astype(np.uint8)
img_pre = img_pre * 255

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (473, 473))

cv2.imshow('', img)
cv2.waitKey()
cv2.imshow('', mask * 255)
cv2.waitKey()
cv2.imshow('', img_pre)
cv2.waitKey()

