import numpy as np
from numba import njit, float64  # type: ignore


@njit(float64(float64[:], float64[:]), cache=True)
def cost(a: np.ndarray, b: np.ndarray) -> np.float64:
    return np.sum(np.abs(a - b))
