from ..dtw.classical import ClassicalDTW
from ..mputils import consume_queue, write_list_to_queue
from ..sharedtypes import (
    DTWType,
    ExtractedFeature,
    ExtractedFeatureQueue,
    FollowerOutputQueue,
    ModeType,
)
from ..dtw.oltw import OLTW
from ..eprint import eprint
from typing import Callable, Dict, List


class Follower:
    def __init__(
        self,
        # user args
        mode: ModeType,
        dtw: DTWType,
        # OLTW settings
        max_run_count: int,
        search_window: int,
        # output queue
        follower_output_queue: FollowerOutputQueue,
        # Performance and Score info
        P_queue: ExtractedFeatureQueue,
        S: List[ExtractedFeature],
        w_a: float,
        w_b: float,
        w_c: float,
    ):
        self.mode = mode
        self.dtw = dtw
        self.max_run_count = max_run_count
        self.search_window = search_window
        self.follower_output_queue = follower_output_queue
        self.P_queue = P_queue
        self.S = S
        self.w_a = w_a
        self.w_b = w_b
        self.w_c = w_c

        follower_start_map: Dict[ModeType, Dict[DTWType, Callable[[], None]]] = {
            "online": {
                "oltw": self.__start_oltw,
            },
            "offline": {
                "oltw": self.__start_oltw,
                "classical": self.__start_classical,
            },
        }
        dtwtype_start_map = follower_start_map.get(self.mode)
        if dtwtype_start_map is None:
            raise ValueError(f"Unknown mode: {self.mode}")
        self.__start = dtwtype_start_map.get(self.dtw)
        if self.__start is None:
            raise ValueError(
                f"Invalid or unknown combination of mode {self.mode} and dtw type {self.dtw}"
            )
        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        self.__start()
        self.__log("Finished")

    def __start_oltw(self):
        oltw = OLTW(
            self.P_queue,
            self.S,
            self.follower_output_queue,
            self.max_run_count,
            self.search_window,
            self.w_a,
            self.w_b,
            self.w_c,
        )
        oltw.dtw()

    def __start_classical(self):
        P: List[ExtractedFeature] = consume_queue(self.P_queue)
        classical = ClassicalDTW(P, self.S, self.w_a, self.w_b, self.w_c)
        dtw_elems = classical.dtw()
        write_list_to_queue(dtw_elems, self.follower_output_queue)

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
