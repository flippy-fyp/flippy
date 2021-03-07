import multiprocessing as mp
import numpy as np

class OLTW:
    def __init__(self, P_queue: mp.Queue[np.ndarray], S: np.ndarray):
        if len(S.shape) != 2:
            raise ValueError(f"S must be a 2D ndarray, got shape: {S.shape}")
        if len(S) == 0:
            raise ValueError(f"Empty S")

    
        self.S = S  # score (rows)
        