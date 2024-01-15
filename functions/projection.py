import numpy as np
import cv2

def project_pixels_to_color_plane(image, u):
    """
    射影された画像を返す関数。

    :param image: 入力画像（CIE L*a*b* 色空間）
    :param u: 色平面の法線ベクトル
    :return: 射影された画像
    """
    # 画像の形状を取得
    height, width, _ = image.shape

    # 射影された画像を格納するための配列を初期化
    projected_image = np.zeros_like(image)

    # 各画素に対して射影を行う
    for i in range(height):
        for j in range(width):
            # 画素の色ベクトルを取得
            color_vector = image[i, j, :]

            # 色ベクトルを色平面に射影
            projected_vector = color_vector - np.dot(color_vector, u) * u

            # 射影された色ベクトルを保存
            projected_image[i, j, :] = projected_vector

    return projected_image

def rotate_projected_image_to_angle(image, final_angle):
    """
    射影された画像を指定された最終角度に回転させる関数。

    :param image: 入力画像（CIE L*a*b* 色空間）
    :param final_angle: 最終的な角度（度単位）
    :return: 回転された画像
    """
    # 角度をラジアンに変換
    angle_rad = np.radians(final_angle)

    # 回転行列を定義（a*とb*の平面内での回転）
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a, cos_a]])

    # 回転された画像を格納するための配列を初期化
    rotated_image = np.zeros_like(image)

    # 各画素に対して回転を適用
    height, width, channels = image.shape
    for i in range(height):
        for j in range(width):
            L, a, b = image[i, j, :]
            ab = np.array([a, b])
            rotated_ab = rotation_matrix @ ab
            rotated_image[i, j, :] = [L, rotated_ab[0], rotated_ab[1]]

    return rotated_image

# 画像の読み込み
image = cv2.imread('/Users/hiyori/kang/images/Chart26.ppm')
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

height, width, channels = image.shape

lab_process = np.zeros_like(image)
lab_out = np.zeros_like(image)

for i in range(height):
    for j in range(width):
        l_process = 100 * lab_image[i,j,0] / 255
        a_process = lab_image[i,j,1] - 128
        b_process = lab_image[i,j,2] - 128
        lab_process[i, j, :] = [l_process, a_process, b_process]
        # print(lab_process)

# 法線ベクトル u を定義（例として）
u = np.array([ 0. ,-0.07, 0.99])

# 画像の画素を色平面に射影
projected_image = project_pixels_to_color_plane(lab_process, u)

rotate_image = rotate_projected_image_to_angle(projected_image, 90 + 11.48)

for i in range(height):
    for j in range(width):
        l_out = 255 * rotate_image[i,j,0] / 100
        a_out = rotate_image[i,j,1] + 128
        b_out = rotate_image[i,j,2] + 128
        lab_out[i, j, :] = [l_out, a_out, b_out]
        # print(lab_process)

img_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

# 射影された画像を表示
cv2.imshow('rotate_image', img_out)
# cv2.imwrite('/Users/hiyori/kang/images/Chart26_kang.ppm',cv2.cvtColor(rotate_image, cv2.COLOR_LAB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
