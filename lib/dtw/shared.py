import numpy as np


def cost(a: np.ndarray, b: np.ndarray) -> np.float64:
    return np.sum(np.abs(a - b))
