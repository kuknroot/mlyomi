# %%
# 制作日時
# 2024/08/08
# 参考文献
# [1] https://www.jstage.jst.go.jp/article/pjsai/JSAI2016/0/JSAI2016_3L4OS16b2/_pdf/-char/ja

import numpy as np
from typing import Union, List

def calc_distance_matrix(X: Union[np.ndarray, List[List[float]]], method: str) -> np.ndarray:
    """
    行列Xの各行間の距離を計算し、距離行列を返す関数。

    :param X: データセットを表す行列（np.ndarrayまたはlist[list[float]]）
    :param method: 距離の計算方法（'euclidean', 'manhattan', 'cosine'）
    :return: 距離行列
    """
    X = np.array(X)
    num_rows = X.shape[0]
    distance_matrix = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(num_rows):
            if i != j:
                distance_matrix[i, j] = calc_distance(X[i], X[j], method)
    
    return distance_matrix

def calc_distance(v: np.ndarray, w: np.ndarray, method: str) -> float:
    """
    2つのデータ間の距離を計算する関数。

    :param v: 1つ目のデータセット（np.ndarray）
    :param w: 2つ目のデータセット（np.ndarray）
    :param method: 距離の計算方法（'euclidean', 'manhattan', 'cosine'）
    :return: データ間の距離
    """
    match method:
        case 'euclidean':
            # ユークリッド距離の計算
            distance: float = np.linalg.norm(v - w)
        case 'manhattan':
            # マンハッタン距離の計算
            distance: float = np.sum(np.abs(v - w))
        case 'cosine':
            # コサイン類似度の計算
            dot_product: float = np.dot(v, w)
            norm_v: float = np.linalg.norm(v)
            norm_w: float = np.linalg.norm(w)
            distance: float = 1 - (dot_product / (norm_v * norm_w))
        case _:
            raise ValueError(f"Unsupported method: {method}")
    
    return distance

def calc_edge_density(
        X: np.ndarray, # distance_matrix
        threshold: float, # d
    ) -> float:
    N: int = X.shape[0]
    edge_matrix: np.ndarray =  X <= threshold
    edge_density: float = (edge_matrix.sum() - N) / (N * (N-1))

    return edge_density

# 例としてデータセットを作成
X: np.ndarray = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 各距離の計算例
euclidean_distance_matrix: np.ndarray = calc_distance_matrix(X, 'euclidean')
manhattan_distance_matrix: np.ndarray = calc_distance_matrix(X, 'manhattan')
cosine_distance_matrix: np.ndarray = calc_distance_matrix(X, 'cosine')

print("ユークリッド距離行列:\n", euclidean_distance_matrix)
print("マンハッタン距離行列:\n", manhattan_distance_matrix)
print("コサイン距離行列:\n", cosine_distance_matrix)
