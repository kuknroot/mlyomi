# %%
import numpy as np
import pandas as pd

# %%
from exec_idtw_kmedoids import KMedoids

# %%
# 分類したいクラスタ数をいろいろ変えて試してみる
n_cluster = 5
x_data = np.ndarray()

# k-meansと同じように初期化ではクラスタ数を指定する
km = KMedoids(n_cluster=n_cluster)
calc_idtw

# k-medoidsの利点として、座標がなくても距離行列があればクラスタリングができるので
# (k-meansの場合は座標が必要)行列を入力データとしている
# そのため、データを行列に変えている
D = squareform(pdist(x_data, metric=))
predicted_labels = km.fit_predict(D)
centroids = km.cluster_centers_

# %%
def estimate_black_litterman(
    degree_of_investor_confidence: float,
    risk_aversion: float,
    init_weights: np.ndarray,
    return_covariance_matrix: np.ndarray,  
) -> tuple[np.ndarray, np.ndarray]:
    pass

df = pd.read_csv("./data/toyota.csv")
df["Adj Close"].values
t = df["Adj Close"].values
length: int = 100
np.array([t[i:i+length] for i, x in enumerate(t[:-length])])
np.array([2*int(t[i+length] > t[i+length-1])-1 for i, x in enumerate(t[:-length])])