import numpy as np
import cv2
from math import sqrt, atan2, pi
from functions import local_contrast
from functions import global_contrast
from mip import Model, maximize, minimize, xsum



image = cv2.imread('/Users/hiyori/kang/images/dot.png')
assert image is not None, "読み込みに失敗しました"

result_image = image
alpha = 0.5

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


#L(u)を計算する関数
l = local_contrast.local_contrast(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
# print(l)

#G(u)を計算する関数
# g = global_contrast.global_contrast(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))

#J(u)を計算する関数
# j = alpha * l + (1 - alpha) * g

m = Model()  # 数理モデル
x = m.add_var("u")

m.objective = minimize(100 * x)

m += x >= 10 #制約条件
m += x <= 100

m.optimize()  # ソルバーの実行

# print(x.x)

#最終的な画像を出す
print("done!")
# cv2.imshow('result', result_image)
# cv2.waitKey()
# cv2.destroyAllWindows()