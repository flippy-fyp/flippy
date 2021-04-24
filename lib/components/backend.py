from ..eprint import eprint
from typing import Callable, Iterator, List, Dict, Set
from ..sharedtypes import (
    FollowerOutputQueue,
    ModeType,
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
        mode: ModeType,
        backend: BackendType,
        follower_output_queue: FollowerOutputQueue,
        performance_stream_start_conn: MultiprocessingConnection,
        score_note_onsets: List[NoteInfo],
        hop_len: int,
        frame_len: int,
        backend_compensation: bool,
        sample_rate: int,
        backend_output: str,
        backend_backtrack: bool,
    ):
        self.mode = mode
        self.follower_output_queue = follower_output_queue
        self.performance_stream_start_conn = performance_stream_start_conn
        self.hop_len = hop_len
        self.frame_len = frame_len
        self.backend_compensation = backend_compensation
        self.sample_rate = sample_rate
        self.backend_backtrack = backend_backtrack

        self.__sorted_note_onsets: SortedDict[float, NoteInfo] = get_sorted_note_onsets(
            score_note_onsets
        )

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
        elif len(backend_output) > 4 and backend_output[:4] == "udp:":
            backend_output_without_udp_scheme = backend_output[4:]
            # try to parse UDP IP and port
            address_port = backend_output_without_udp_scheme.split(":")
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
        else:
            # truncate the file first
            with open(backend_output, "w+"):
                pass

            def p(x: str):
                with open(backend_output, "a") as f:
                    f.write(f"{x}\n")

            self.__output_func = p

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        self.__start()
        self.__log("Finished")

    def __start_timestamp(self):
        prev_s = -1
        while True:
            e = self.follower_output_queue.get()
            if e is None:
                return
            s = e[1]
            if (self.backend_backtrack and s != prev_s) or (
                not self.backend_backtrack and s > prev_s
            ):
                timestamp_s = self.__get_online_timestamp(s)
                self.__output_func(timestamp_s)
                prev_s = s

    def __get_online_timestamp(self, s: int) -> float:
        if self.backend_compensation:
            return float(self.frame_len + (s - 1) * self.hop_len) / self.sample_rate
        return float(self.hop_len * s) / self.sample_rate

    def __start_alignment(self):
        # wait for the performance stream to start
        performance_start_time: float = self.performance_stream_start_conn.recv()
        prev_s = -1
        seen_closest_notes_time: Set[float] = set()

        while True:
            e = self.follower_output_queue.get()
            if e is None:
                return
            p, s = e
            if (self.backend_backtrack and s != prev_s) or (
                not self.backend_backtrack and s > prev_s
            ):
                prev_s = s
                curr_time = time.perf_counter()
                det_time_ms = (curr_time - performance_start_time) * 1000
                # use ms because NoteInfo are ms and the follower output for quantitative
                # testbench is ms
                timestamp_p_s = float(self.hop_len * p) / self.sample_rate
                timestamp_p_ms = timestamp_p_s * 1000
                timestamp_s_s = float(self.hop_len * s) / self.sample_rate
                timestamp_s_ms = timestamp_s_s * 1000

                closest_notes = get_closest_notes_before(
                    self.__sorted_note_onsets, timestamp_s_ms
                )
                if len(closest_notes) == 0:
                    # self.__log(f"Ignoring unfound closest notes at {timestamp_s_ms}")
                    pass
                elif closest_notes[0].note_start not in seen_closest_notes_time:
                    seen_closest_notes_time.add(closest_notes[0].note_start)
                    for closest_note in closest_notes:
                        if self.mode == "online":
                            # MIREX format
                            self.__output_func(
                                f"{round(timestamp_p_ms)} {round(det_time_ms)} {round(closest_note.note_start)} {closest_note.midi_note_num}"
                            )
                        elif self.mode == "offline":
                            # put det_time as est_time
                            self.__output_func(
                                f"{round(timestamp_p_ms)} {round(timestamp_p_ms)} {round(closest_notes[0].note_start)} {closest_note.midi_note_num}"
                            )
                        else:
                            raise ValueError(f"Unknown mode: {self.mode}")

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")


def get_sorted_note_onsets(
    score_note_onsets: List[NoteInfo],
) -> "SortedDict[float, List[NoteInfo]]":
    # group by note_start time
    res: "SortedDict[float, List[NoteInfo]]" = SortedDict()

    for onset in score_note_onsets:
        if onset.note_start not in res:
            res[onset.note_start] = []
        res[onset.note_start].append(onset)

    return res


def get_closest_notes_before(
    sorted_note_onsets: "SortedDict[float, List[NoteInfo]]", timestamp_ms: float
) -> List[NoteInfo]:
    closest_note_time_before_generator: Iterator[float] = sorted_note_onsets.irange(
        maximum=timestamp_ms, reverse=True
    )

    closest_notes_before_key = next(closest_note_time_before_generator, None)
    return sorted_note_onsets.get(closest_notes_before_key, [])
