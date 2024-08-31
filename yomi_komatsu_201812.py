# %%
# 説明は以下
# 

# %%
import numpy as np

# %%
def calc_return(price_vec: np.ndarray, dividend_vec: np.ndarray | None = None) -> np.ndarray:
    """
    # 概要
    価格ベクトルと配当(あれば)からリターンを求める関数を実装する

    # 参考文献
    [1] https://www.asakura.co.jp/detail.php?book_code=27585
    - 数式(1.1)
    """
    if dividend_vec is None:
        return (price_vec[1:] - price_vec[:-1]) / price_vec[:-1]
    else:
        return (price_vec[1:] + dividend_vec[1:] - price_vec[:-1]) / price_vec[:-1]

# %%
def calc_return_mean(return_vec: np.ndarray) -> float:
    """
    # 概要
    リターンベクトルの平均を求める関数を実装する。

    # 参考文献
    [1] https://www.asakura.co.jp/detail.php?book_code=27585
    - 数式(1.2)
    """
    return return_vec.mean()

def calc_return_var(return_vec: np.ndarray) -> float:
    """
    # 概要
    リターンベクトルの平均を求める分散を実装する。
    - ポイントは、np.varのddofの値が1であること。

    # 参考文献
    [1] https://www.asakura.co.jp/detail.php?book_code=27585
    - 数式(1.3)
    """
    return np.var(return_vec, ddof=1)

def calc_return_std(return_var: float) -> float:
    """
    # 概要
    リターンベクトルの不偏分散から標準偏差を計算する。

    # 参考文献
    [1] https://www.asakura.co.jp/detail.php?book_code=27585
    - 数式(1.4)
    """
    return np.sqrt(return_var)

def calc_return_means_covmatrix(return_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    # 概要
    複数のリターンベクトルの系列から平均ベクトルと分散共分散行列を計算する。

    # 参考文献
    [1] https://www.asakura.co.jp/detail.php?book_code=27585
    - 数式(1.5),(1.6),(1.7),(1.8)
    """
    return return_matrix.mean(axis=1), np.cov(return_matrix, rowvar=True)

def calc_portfolio_means_covmatrix(weight_vec: np.ndarray, return_matrix: np.ndarray) -> np.ndarray:
    """
    # 概要
    複数のリターンベクトルの系列とポートフォリオ加重から、ポートフォリオの平均ベクトルと分散共分散行列を計算する。

    # 参考文献
    [1] https://www.asakura.co.jp/detail.php?book_code=27585
    - 数式(1.5),(1.6),(1.7),(1.8)
    """
    mean_vec, var_vec: tuple[np.ndarray, np.ndarray] = calc_return_means_covmatrix(return_matrix)

    return return_matrix.mean(axis=1), np.cov(return_matrix, rowvar=True)

def row_covariance_matrix(matrix):
    """
    np.ndarrayの行列から、各行同士の分散共分散行列を求める関数。

    Parameters:
    matrix (np.ndarray): 入力行列

    Returns:
    np.ndarray: 各行同士の分散共分散行列
    """
    return np.cov(matrix, rowvar=True)