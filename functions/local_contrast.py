import numpy as np

def local_contrast(image, sigma):
    if sigma <= 0:
        raise ValueError("sigma must be a positive value")

    height, width, _ = image.shape
    N = height * width
    Xl = np.zeros((N, 3))  # 色差行列の初期化

    # 画像をフラット化して処理しやすくする
    flat_image = image.reshape(N, 3)

    # 各ピクセルに対してランダムな近傍ピクセルを選択
    for i in range(N):
        # ピクセルiの位置を計算
        x, y = i % width, i // width
        # 近傍ピクセルの座標をランダムに選択
        nx, ny = min(max(int(x + np.random.normal(0, sigma)), 0), width - 1), min(max(int(y + np.random.normal(0, sigma)), 0), height - 1)
        neighbor_index = ny * width + nx
        # 色差ベクトルを計算
        Xl[i, :] = flat_image[i, :] - flat_image[neighbor_index, :]

    # 局所コントラストを計算
    local_contrast_matrix = np.dot(Xl, Xl.T) / N

    return local_contrast_matrix
