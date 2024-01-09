import numpy as np
import cv2
from math import sqrt, atan2, pi
from functions import local_contrast
from functions import global_contrast
from functions import calculate_color_difference_vectors_with_gaussian_pairing
from scipy.optimize import minimize

image = cv2.imread('/Users/hiyori/kang/images/Lena.ppm')
assert image is not None, "読み込みに失敗しました"

height, width, _ = image.shape
N = height * width

result_image = image
alpha = 1.0

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


#L(u)を計算する関数
l = local_contrast.local_contrast(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))

#TODO G(u)を計算する関数
# g = global_contrast.global_contrast(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))

#TODO J(u)を計算する関数
# j = alpha * l + (1 - alpha) * g

#Xlを計算する関数
Xl = calculate_color_difference_vectors_with_gaussian_pairing.calculate_color_difference_vectors_with_gaussian_pairing(image)
Al = Xl.T @ Xl

# 目的関数と制約条件の定義
def objective(u):
    return u.T @ Al @ u

# u = m.add_var("u")
def constraint(u):
    return np.dot(u, u) - 1

# L*軸に垂直な制約
def constraint_perpendicular_to_L_star(u):
    e = np.array([1, 0, 0])  # L*軸を指す標準基底ベクトル
    return np.dot(u, e)

# 単位ベクトル制約
def constraint_unit_vector(u):
    return np.dot(u, u) - 1

# 初期値の設定
u0 = np.random.rand(3)
u0 /= np.linalg.norm(u0) #単位ベクトル化

# 制約の設定
constraints = [{'type': 'eq', 'fun': constraint_perpendicular_to_L_star},
               {'type': 'eq', 'fun': constraint_unit_vector}]

# 最適化問題の解決
res = minimize(objective, u0, constraints=constraints)

# 最適化された u の値
optimized_u = res.x

# 最適化された結果の u^T Al u の値を計算
optimized_value = (optimized_u.T @ Al @ optimized_u) / N

# 結果の表示
print("Optimized u:", res.x)
print("Optimized value of u.T Al u:", optimized_value)

#TODO 最終的な画像を出す
print("done!")
# cv2.imshow('result', result_image)
# cv2.waitKey()
# cv2.destroyAllWindows()