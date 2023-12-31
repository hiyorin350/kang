import numpy as np

def calculate_color_difference_vectors_with_gaussian_pairing(image):
    """
    ガウシアンペアリングを使用して画像内の各ピクセルの色差ベクトルを計算します。

    :param image: CIE L*a*b* 色空間の入力画像。
    :param sigma: ペアリングに使用されるガウス分布の標準偏差。
    :return: 色差ベクトルの配列。
    sigmaは分散ではなく標準偏差であることに注意
    """
    height, width, _ = image.shape
    N = height * width
    Xl = np.zeros((N, 3))  # 色差行列の初期化

    sigma = np.sqrt((2 / np.pi) * np.sqrt(2 * min(height, width)))

    # 画像をフラット化して処理しやすくする
    flat_image = image.reshape(N, 3)

    # ガウス分布を使用してランダムなペアを生成
    for i in range(N):
        # ピクセルの座標を計算
        x, y = i % width, i // width

        # ガウス分布に基づいてランダムなオフセットを生成
        offset_x, offset_y = np.random.normal(0, sigma, 2)
        nx, ny = int(x + offset_x), int(y + offset_y)

        # 新しい座標が画像の範囲内に収まるようにする
        nx = max(0, min(nx, width - 1))
        ny = max(0, min(ny, height - 1))

        # 色差ベクトルを計算
        neighbor_index = ny * width + nx
        Xl[i, :] = flat_image[i, :] - flat_image[neighbor_index]

    return Xl
