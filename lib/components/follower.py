import multiprocessing as mp
import numpy as np


class Follower:
    def __init__(
        self,
        mode: str,
        dtw: str,
        max_run_count: int,
        search_window: int,
        P_queue: "mp.Queue[np.ndarray]",
        S: np.ndarray,
    ):
        self.mode = mode
        self.dtw = dtw
        self.max_run_count = max_run_count
        self.search_window = search_window
        self.P_queue = P_queue
        self.S = S

        if mode == "online":
            raise NotImplementedError("Online mode not yet implemented!")
        elif mode == "offline":
            pass
        else:
            raise ValueError(f"Unknown mode {mode}")
        pass
