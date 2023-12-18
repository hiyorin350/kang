import numpy as np
import cv2
from math import sqrt, atan2, pi

image = cv2.imread('images/Lena.ppm')
assert image is not None, "読み込みに失敗しました"

result_image = image

#L(u)を計算する関数

#G(u)を計算する関数

#J(u)を計算する関数

#最終的な画像を出す
cv2.imshow('Lena', result_image)
cv2.waitKey()
cv2.destroyAllWindows()