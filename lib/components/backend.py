from lib.eprint import eprint
from typing import Iterator, List, Optional
from lib.sharedtypes import DTWPathElemType, NoteInfo, BackendType
import multiprocessing as mp
from lib.constants import DEFAULT_SAMPLE_RATE
import time
import math
from sortedcontainers import SortedDict  # type: ignore


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

        self.__sorted_note_onsets: SortedDict[float, NoteInfo] = SortedDict(
            {x.note_start: x for x in self.note_onsets}
        )  # note_start is ms

    def start(self):
        if self.backend == "timestamp":
            self.__start_timestamp()
        elif self.backend == "alignment":
            self.__start_alignment()

    def __start_timestamp(self):
        prev_s = -1
        while True:
            e = self.follower_output_queue.get()
            if e is None:
                return
            s = e[1]
            if s != prev_s:
                timestamp_s = float(self.slice_len * s) / self.sample_rate
                print(timestamp_s)
                prev_s = s

    def __start_alignment(self):
        # wait for the performance stream to start
        performance_start_time: float = self.performance_stream_start_conn.recv()
        prev_s = -1
        prev_note_info: Optional[NoteInfo] = None

        while True:
            e = self.follower_output_queue.get()
            if e is None:
                return
            s = e[1]
            p = e[2]
            if s != prev_s:
                curr_time = time.perf_counter()
                det_time = curr_time - performance_start_time
                # use ms because NoteInfo are ms and the follower output for quantitative
                # testbench is ms
                timestamp_s_s = float(self.slice_len * s) / self.sample_rate
                timestamp_s_ms = timestamp_s_s * 1000
                timestamp_p_s = float(self.slice_len * p) / self.sample_rate
                timestamp_p_ms = timestamp_p_s * 1000

                closest_note = get_closest_note(
                    self.__sorted_note_onsets, timestamp_s_ms
                )
                if closest_note is None:
                    self.__log("Ignoring unfound closest note!")
                if closest_note != prev_note_info:
                    # MIREX format
                    print(
                        f"{timestamp_p_ms} {det_time} {closest_note.note_start} {closest_note.midi_note_num}"
                    )
                prev_note_info = closest_note

    def __log(self, *args, **kwargs):
        eprint(*args, **kwargs)


def get_closest_note(
    sorted_note_onsets: "SortedDict[float, NoteInfo]", timestamp_ms: float
) -> Optional[NoteInfo]:
    closest_note_after_generator: Iterator[float] = sorted_note_onsets.irange(
        minimum=timestamp_ms
    )
    closest_note_after_timestamp_ms = next(closest_note_after_generator, None)
    gap_to_closest_note_after = (
        math.inf
        if closest_note_after_timestamp_ms is None
        else closest_note_after_timestamp_ms - timestamp_ms
    )

    closest_note_before_generator: Iterator[float] = sorted_note_onsets.irange(
        maximum=timestamp_ms, reverse=True
    )

    closest_note_before_timestamp_ms = next(closest_note_before_generator, None)
    gap_to_closest_note_before = (
        math.inf
        if closest_note_before_timestamp_ms is None
        else timestamp_ms - closest_note_before_timestamp_ms
    )

    if gap_to_closest_note_before == math.inf and gap_to_closest_note_after == math.inf:
        # reject if none found, should not happen
        return None

    if gap_to_closest_note_before <= gap_to_closest_note_after:
        return sorted_note_onsets[closest_note_before_timestamp_ms]
    else:
        return sorted_note_onsets[closest_note_after_timestamp_ms]
