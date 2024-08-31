import numpy as np

from enum import Enum
from collections.abc import Callable as callable, Iterable as iterable
from typing import NoReturn

realnumber = int | float

class LossEnum(str, Enum):
    """
    # 概要
    どの損失関数を利用するかを選択するEnum
    re: relative entropy
    se: square error
    hd: hellinger distance
    ae: absolute error

    # 参考情報
    [1] https://www.jstage.jst.go.jp/article/jsaisigtwo/2010/DMSM-A903/2010_10/_pdf
    """

    re = "re"
    se = "se"
    hd = "hd"
    ae = "ae"

class WeightEnum(str, Enum):
    """
    # 概要
    どの重み付けアルゴリズムを利用するかを選択するEnum
    ## WAA
    Weighted Averaging Algorithm
    重み付き平均アルゴリズム
    ## WW
    Weighted Window
    ## WWH
    Weighted Window with follow the leading History

    # 参考情報
    [1] https://www.jstage.jst.go.jp/article/jsaisigtwo/2010/DMSM-A903/2010_10/_pdf
    """

    WAA = "waa"
    WW = "ww"
    WWH = "wwh"

def check_loss_constant(
        loss_constant: float,
        loss: LossEnum
    ) -> None | NoReturn:
    """
    # 概要
    損失関数の種類と定数を与えて、定数が損失関数的に正しい範囲に収まっているか確認する。
    1. 満たさなかったらエラーを返す
    2. 満たせば何もしない

    # 参考情報
    [1] https://www.jstage.jst.go.jp/article/jsaisigtwo/2010/DMSM-A903/2010_10/_pdf
    [2] https://www.kspub.co.jp/book/detail/1529229.html
    """

    match loss:
        case LossEnum.re:
            if 0 <= loss_constant and loss_constant <= 1:
                pass
            else:
                raise ValueError("定数エラー")
        case LossEnum.se:
            if 0 <= loss_constant and loss_constant <= .5:
                pass
            else:
                raise ValueError("定数エラー")
        case LossEnum.hd:
            #[1]では最大でも2 ** 0.5、[2]では1にしろという設定だったので1にしました。
            if 0 <= loss_constant and loss_constant <= 1:
                pass
            else:
                raise ValueError("定数エラー")
        case LossEnum.ae:
            if 0 <= loss_constant:
                pass
            else:
                raise ValueError("定数エラー")
        case _:
            raise ValueError("Unknown loss function")

def calc_loss(
        loss: LossEnum,
        y_pred: np.ndarray,
        y_data: realnumber,
    ) -> np.ndarray:
    """
    # 概要
    各予測に対する損失関数を計算する。

    # 参考情報
    [1] https://www.jstage.jst.go.jp/article/jsaisigtwo/2010/DMSM-A903/2010_10/_pdf
    [2] https://www.kspub.co.jp/book/detail/1529229.html
    """

    match loss:
        case LossEnum.re:
            epsilon: realnumber = 1e-15
            y_data: realnumber = np.clip(y_data, epsilon, 1 - epsilon)
            y_pred: np.ndarray = np.clip(y_pred, epsilon, 1 - epsilon)
            return y_data * np.log(y_data / y_pred) + (1 - y_data) * np.log((1 - y_data) / (1 - y_pred))
        case LossEnum.se:
            return (y_pred - y_data) ** 2
        case LossEnum.hd:
            y_data: realnumber = np.clip(y_data, 0, 1)
            y_pred: np.ndarray = np.clip(y_pred, 0, 1)
            term_1_y: np.ndarray = np.sqrt(1 - y_pred) - np.sqrt(1 - y_data)
            term_y: np.ndarray = np.sqrt(y_pred) - np.sqrt(y_data)
            return 0.5 * (term_1_y ** 2 + term_y ** 2)
        case LossEnum.ae:
            return np.abs(y_pred - y_data)
        case _:
            raise ValueError("Unknown loss function")

def update_weights(
        init_weight: np.ndarray,
        weight_algo: WeightEnum,
        loss_constant: float,
        loss: LossEnum,
        y_pred: np.ndarray,
        y_data: realnumber,
    ) -> np.ndarray:
    """
    # 概要
    以下をやる
    1. calc_lossによる損失関数計算
    2. それを使って損失関数を計算する

    # 参考情報
    [1] https://www.jstage.jst.go.jp/article/jsaisigtwo/2010/DMSM-A903/2010_10/_pdf
    """

    # 損失関数を計算する
    np_loss: np.array = calc_loss(loss, y_pred, y_data)

    # ウェイトをアップデート
    match weight_algo:
        case WeightEnum.WAA:
            np_weight: np.ndarray = init_weight * np.exp(loss_constant * np_loss)
            return np_weight / np_weight.sum()
        case WeightEnum.WW:
            np_weight: np.ndarray = np.exp(loss_constant * np_loss)
            return np_weight / np_weight.sum()
        case WeightEnum.WWH:
            np_weight: np.ndarray = init_weight * np.exp(loss_constant * np_loss)
            return np_weight / np_weight.sum()
        case _:
            raise ValueError("Unknown weight algoithm")

def update_weights(
        init_weight: np.ndarray,
        weight_algo: WeightEnum,
        loss_constant: float,
        loss: LossEnum,
        y_pred: np.ndarray,
        y_data: realnumber,
    ) -> np.ndarray:
    """
    # 概要
    以下をやる
    1. calc_lossによる損失関数計算
    2. それを使って損失関数を計算する

    # 参考情報
    [1] https://www.jstage.jst.go.jp/article/jsaisigtwo/2010/DMSM-A903/2010_10/_pdf
    """

    # 損失関数を計算する
    np_loss: np.array = calc_loss(loss, y_pred, y_data)

    # ウェイトをアップデート
    match weight_algo:
        case WeightEnum.WAA:
            np_weight: np.ndarray = init_weight * np.exp(loss_constant * np_loss)
            return np_weight / np_weight.sum()
        case WeightEnum.WW:
            np_weight: np.ndarray = init_weight * np.exp(loss_constant * np_loss)
            return np_weight / np_weight.sum()
        case WeightEnum.WWH:
            np_weight: np.ndarray = init_weight * np.exp(loss_constant * np_loss)
            return np_weight / np_weight.sum()
        case _:
            raise ValueError("Unknown weight algoithm")
```