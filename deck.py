# %%
import numpy as np

# %%
def estimate_black_litterman(
    degree_of_investor_confidence: float,
    risk_aversion: float,
    init_weights: np.ndarray,
    return_covariance_matrix: np.ndarray,
    
    
) -> tuple[np.ndarray, np.ndarray]:
    