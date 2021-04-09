from lib.components.backend import Backend
from lib.sharedtypes import BackendType, DTWType, ModeType, NoteInfo
from lib.dtw.oltw import OLTW
from lib.eprint import eprint
import multiprocessing as mp
import numpy as np
import time
from typing import Optional, List


class Follower:
    def __init__(
        self,
        # user args
        mode: ModeType,
        dtw: DTWType,
        backend: BackendType,
        # OLTW settings
        max_run_count: int,
        search_window: int,
        # other settings
        slice_len: int,
        sample_rate: int,
        # Performance and Score info
        P_queue: "mp.Queue[np.ndarray]",
        S: np.ndarray,
        note_onsets: List[NoteInfo],
    ):
        self.mode = mode
        self.dtw = dtw
        self.backend = backend
        self.max_run_count = max_run_count
        self.search_window = search_window
        self.P_queue = P_queue
        self.S = S
        self.note_onsets = note_onsets
        self.slice_len = slice_len
        self.sample_rate = sample_rate

    def start(self):
        if self.mode == "online":
            if self.dtw == "classical":
                raise ValueError("classical dtw cannot be used in online mode")
            elif self.dtw == "oltw":
                eprint("online oltw mode")

                output_queue: "np.Queue[Optional[np.ndarray]]" = mp.Queue()
                oltw = OLTW(
                    self.P_queue,
                    self.S,
                    output_queue,
                    self.max_run_count,
                    self.search_window,
                )
                performance_stream_start_conn: mp.connection.Connection = (
                    mp.connection.Connection()
                )

                backend = Backend(
                    self.backend,
                    output_queue,
                    performance_stream_start_conn,
                    self.note_onsets,
                    self.slice_len,
                    self.sample_rate,
                )
                # Start
                backend.start()
                oltw.start()
                performance_stream_start_conn.send(time.perf_counter())

            else:
                raise ValueError(f"Unknown dtw: {self.mode}")
        elif self.mode == "offline":
            if self.dtw == "classical":
                raise ValueError("classical dtw cannot be used in online mode")
            elif self.dtw == "oltw":
                pass
            else:
                raise ValueError(f"Unknown dtw: {self.mode}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        pass
