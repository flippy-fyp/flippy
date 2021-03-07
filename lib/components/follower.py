import multiprocessing as mp
import numpy as np


class Follower:
    def __init__(
        self,
        mode: str,
        dtw: str,
        max_run_count: int,
        search_window: int,
        p_queue: mp.Queue[np.ndarray],
        s_queue: mp.Queue[np.ndarray],
    ):
        self.mode = mode
        self.dtw = dtw
        self.max_run_count = max_run_count
        self.search_window = search_window
        self.p_queue = p_queue
        self.s_queue = s_queue

        if mode == "online":
            raise NotImplementedError("Online mode not yet implemented!")
        pass

