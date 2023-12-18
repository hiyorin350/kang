import numpy as np
import cv2
from math import sqrt, atan2, pi
from functions import local_contrast
from functions import global_contrast

image = cv2.imread('/Users/hiyori/kang/images/dot.png')
assert image is not None, "読み込みに失敗しました"

result_image = image
sigma = 2
alpha = 0.5

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

#L(u)を計算する関数
l = local_contrast.local_contrast(cv2.cvtColor(image, cv2.COLOR_BGR2LAB), sigma)
print(l)

#G(u)を計算する関数
g = global_contrast.global_contrast(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))

#J(u)を計算する関数
j = alpha * l + (1 - alpha) * g

#最終的な画像を出す
print("done!")
cv2.imshow('result', result_image)
cv2.waitKey()
cv2.destroyAllWindows()