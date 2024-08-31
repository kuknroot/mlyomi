# %%
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from scipy.spatial.distance import pdist, squareform

# %%
from dtw_families import calc_idtw

# %%
class BaseSchema(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

# %%

class KMedoids(BaseSchema):
    n_cluster: int = Field(..., description="Number of clusters")
    max_iter: int = Field(1000, description="Maximum number of iterations")

    def fit_predict(self, D: np.ndarray) -> np.ndarray:
        """
        Fit the K-Medoids model to the data and predict the cluster for each data point.

        Args:
            D (np.ndarray): Distance matrix of shape (m, n).

        Returns:
            np.ndarray: Array of cluster labels for each data point.
        """
        m, n = D.shape

        initial_medoids = np.random.choice(range(m), self.n_cluster, replace=False)
        tmp_D = D[:, initial_medoids]

        # Cluster based on the closest initial medoid
        labels = np.argmin(tmp_D, axis=1)

        # Create a DataFrame with unique IDs for easier handling
        results = pd.DataFrame({'id': range(m), 'label': labels})

        col_names = [f'x_{i + 1}' for i in range(m)]
        results = pd.concat([results, pd.DataFrame(D, columns=col_names)], axis=1)

        before_medoids = initial_medoids.tolist()
        new_medoids = []

        loop = 0
        while len(set(before_medoids).intersection(set(new_medoids))) != self.n_cluster and loop < self.max_iter:
            if loop > 0:
                before_medoids = new_medoids.copy()
                new_medoids = []

            for i in range(self.n_cluster):
                tmp = results[results['label'] == i].copy()
                tmp['distance'] = np.sum(tmp[[f'x_{id + 1}' for id in tmp['id']]].values, axis=1)
                tmp = tmp.reset_index(drop=True)
                new_medoids.append(tmp.loc[tmp['distance'].idxmin(), 'id'])

            new_medoids = sorted(new_medoids)
            tmp_D = D[:, new_medoids]

            clustaling_labels = np.argmin(tmp_D, axis=1)
            results['label'] = clustaling_labels

            loop += 1

        results = results[['id', 'label']]
        results['flag_medoid'] = 0
        for medoid in new_medoids:
            results.loc[results['id'] == medoid, 'flag_medoid'] = 1

        tmp_D = pd.DataFrame(tmp_D, columns=[f'medoid_distance{i}' for i in range(self.n_cluster)])
        results = pd.concat([results, tmp_D], axis=1)

        self.results = results
        self.cluster_centers_ = new_medoids

        return results['label'].values


# %%
# 分類したいクラスタ数をいろいろ変えて試してみる
n_clusters = 10
x_data = np.ndarray()

# k-meansと同じように初期化ではクラスタ数を指定する
km = KMedoids(n_cluster=n_clusters)
calc_idtw

# k-medoidsの利点として、座標がなくても距離行列があればクラスタリングができるので
# (k-meansの場合は座標が必要)行列を入力データとしている
# そのため、データを行列に変えている
D = squareform(pdist(x_data, metric=))
predicted_labels = km.fit_predict(D)
centroids = km.cluster_centers_

# %%
# %%
import numpy as np

# %%
def estimate_black_litterman(
    degree_of_investor_confidence: float,
    risk_aversion: float,
    init_weights: np.ndarray,
    return_covariance_matrix: np.ndarray,
    
    
) -> tuple[np.ndarray, np.ndarray]:
    