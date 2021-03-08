import numpy as np
from numba import njit, float32  # type: ignore


@njit(float32(float32[:], float32[:]), cache=True)
def cost(a: np.ndarray, b: np.ndarray) -> np.float32:
    return np.sum(np.abs(a - b))
