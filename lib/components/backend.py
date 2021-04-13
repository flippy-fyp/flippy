from lib.eprint import eprint
from typing import Callable, Iterator, List, Optional, Dict, Set
from lib.sharedtypes import (
    FollowerOutputQueue,
    NoteInfo,
    BackendType,
    MultiprocessingConnection,
)
import time
from sortedcontainers import SortedDict  # type: ignore
import socket


class Backend:
    """
    Outputs results of alignment
    """

    def __init__(
        self,
        backend: BackendType,
        follower_output_queue: FollowerOutputQueue,
        performance_stream_start_conn: MultiprocessingConnection,
        score_note_onsets: List[NoteInfo],
        slice_len: int,
        sample_rate: int,
        backend_output: str,
    ):
        self.follower_output_queue = follower_output_queue
        self.performance_stream_start_conn = performance_stream_start_conn
        self.slice_len = slice_len
        self.sample_rate = sample_rate

        self.__sorted_note_onsets: SortedDict[float, NoteInfo] = SortedDict(
            {x.note_start: x for x in score_note_onsets}
        )  # note_start is ms

        backend_start_map: Dict[BackendType, Callable[[], None]] = {
            "timestamp": self.__start_timestamp,
            "alignment": self.__start_alignment,
        }
        self.__start = backend_start_map.get(backend)
        if self.__start is None:
            raise ValueError(f"Unknown backend mode: {backend}")

        if backend_output == "stdout":

            def p(x: str):
                print(x, flush=True)

            self.__output_func = p
        elif backend_output == "stderr":

            def p(x: str):
                eprint(x, flush=True)

            self.__output_func = p
        else:
            # try to parse UDP IP and port
            address_port = backend_output.split(":")
            if len(address_port) != 2:
                raise ValueError(f"Unknown `backend_output`: {backend_output}")
            addr = str(address_port[0])
            port = int(address_port[1])

            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.__log(f"Backend will be outputting via UDP to {addr}:{port}")

            def p(x: str):
                eprint(x)
                self.__socket.sendto(str(x).encode(), (addr, port))

            self.__output_func = p

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        self.__start()
        self.__log("Finished")

    def __start_timestamp(self):
        self.__output_func("READY")
        prev_s = -1
        while True:
            e = self.follower_output_queue.get()
            if e is None:
                self.__output_func("END")
                return
            s = e[1]
            if s != prev_s:
                timestamp_s = float(self.slice_len * s) / self.sample_rate
                self.__output_func(timestamp_s)
                prev_s = s

    def __start_alignment(self):
        # wait for the performance stream to start
        performance_start_time: float = self.performance_stream_start_conn.recv()
        prev_s = -1
        seen_notes: Set[NoteInfo] = set()

        while True:
            e = self.follower_output_queue.get()
            if e is None:
                return
            s, p = e
            if s != prev_s:
                curr_time = time.perf_counter()
                det_time_ms = (curr_time - performance_start_time) * 1000
                # use ms because NoteInfo are ms and the follower output for quantitative
                # testbench is ms
                timestamp_s_s = float(self.slice_len * s) / self.sample_rate
                timestamp_s_ms = timestamp_s_s * 1000
                timestamp_p_s = float(self.slice_len * p) / self.sample_rate
                timestamp_p_ms = timestamp_p_s * 1000

                closest_note = get_closest_note_before(
                    self.__sorted_note_onsets, timestamp_s_ms
                )
                if closest_note is None:
                    self.__log("Ignoring unfound closest note!")
                if closest_note not in seen_notes:
                    # MIREX format
                    self.__output_func(
                        f"{int(timestamp_p_ms)} {int(det_time_ms)} {int(closest_note.note_start)} {closest_note.midi_note_num}"
                    )
                    seen_notes.add(closest_note)
                prev_s = s

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")


def get_closest_note_before(
    sorted_note_onsets: "SortedDict[float, NoteInfo]", timestamp_ms: float
) -> Optional[NoteInfo]:
    closest_note_before_generator: Iterator[float] = sorted_note_onsets.irange(
        maximum=timestamp_ms, reverse=True
    )

    closest_note_before_key = next(closest_note_before_generator, None)
    if closest_note_before_key is None:
        return None
    return sorted_note_onsets[closest_note_before_key]
