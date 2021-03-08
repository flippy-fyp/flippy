from typing import List, Optional
from lib.sharedtypes import DTWPathElemType, NoteInfo, BackendType
import multiprocessing as mp
from lib.constants import DEFAULT_SAMPLE_RATE


class Backend:
    """
    Outputs results of alignment
    """

    def __init__(
        self,
        backend: BackendType,
        follower_output_queue: "mp.Queue[Optional[DTWPathElemType]]",
        performance_stream_start_conn: "mp.connection.Connection",
        note_onsets: List[NoteInfo],
        slice_len: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        self.backend = backend
        self.follower_output_queue = follower_output_queue
        self.performance_stream_start_conn = performance_stream_start_conn
        self.note_onsets = note_onsets
        self.slice_len = slice_len
        self.sample_rate = sample_rate

    def start(self):
        # wait for the performance stream to start
        performance_start_time: float = self.performance_stream_start_conn.recv()

        prev_s = -1
        while True:
            e = self.follower_output_queue.get()
            if e is None:
                return

            s = e[1]

            if s != prev_s:
                if self.backend == "timestamp":
                    timestamp_s = self.slice_len * s // self.sample_rate
                    print(timestamp_s)
                elif self.backend == "alignment":
                    raise NotImplementedError("alignment mode not y implemented")

                prev_s = s
