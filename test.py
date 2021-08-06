from model import Unet_1d
import torch
import numpy as np
from loss.bce_loss import BceLoss
from utils.to_onehot import to_one_hot

# x = torch.LongTensor(16, 1, 1280).random_() % 4
# gt_mask = to_one_hot(x, 4)
# logits = torch.rand(16, 4, 1280)
# loss_f = BceLoss()
# logits[gt_mask == 1] = 0.999
# loss = loss_f(logits, gt_mask)
# print(loss.item())

# x = np.random.rand(16, 4, 1280)
# x = np.argmax(x, axis=1)
# print(x.shape)


import cv2 as cv

np.set_printoptions(threshold=np.inf)
binary = np.zeros((100, ))
binary[30:60] = 2
binary[65:100] = 3

binary[10:20] = 0
binary[40:50] = 0
binary[70:90] = 0
print(binary.dtype)
print(binary.shape)
# 核的大小和形状
kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
# 开操作
ret1 = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=5)

print(ret1.shape)
# # 闭操作
ret2 = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=5)
ret2 = ret2.reshape(-1)
print(binary)
print(ret2)
