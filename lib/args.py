from lib.constants import DEFAULT_SAMPLE_RATE
from lib.sharedtypes import ModeType, DTWType, CQTType, BackendType
from lib.utils import quantise_hz_midi
from tap import Tap  # type: ignore
from lib.eprint import eprint
from sys import exit
from os import path
from typing import List


class Arguments(Tap):
    mode: ModeType = "online"  # Mode: `offline` or `online`.
    dtw: DTWType = "oltw"  # DTW Algo: `classical` or `oltw`. `classical` is only available for the `offline` mode.
    cqt: CQTType = "nsgt"  # CQT Algo: `nsgt`, `librosa_pseudo`, `librosa_hybrid` or `librosa`. `librosa` is only available for the `offline` mode.
    max_run_count: int = 3  # `MaxRunCount` for `online` mode with `oltw` DTW.
    search_window: int = 250  # `SearchWindow` for `online` mode with `oltw` DTW.
    fmin: float = 130.8  # Minimum frequency (Hz) for CQT.
    fmax: float = 4186.0  # Maximum frequency (Hz) for CQT.
    slice_len: int = (
        2048  # Slice length for `nsgt` cqt, or hop_length in `librosa` cqt.
    )
    transition_slice_ratio: int = 4  # Transition to slice length ratio for `nsgt` cqt.

    perf_wave_path: str  # Path to performance WAVE file.
    score_midi_path: str  # Path to score MIDI.

    backend: BackendType = (
        "alignment"  # Alignment result type: `alignment` or `timestamp`.
    )

    simulate_performance: bool = (
        False  # Whether to stream performance "live" into the system.
    )

    sample_rate: int = DEFAULT_SAMPLE_RATE  # Sample rate to synthesise score and load performance wave file.

    play_performance_audio: bool = False  # Whether to play the performance audio file when started. Requires `simulate_performance` to be set to True.

    def __log_and_exit(self, msg: str):
        self.__log(f"Argument Error: {msg}.")
        self.__log("Use the `--help` flag to show the help message.")
        exit(1)

    def sanitize(self):
        if self.max_run_count < 0:
            self.__log_and_exit(f"max_run_count must be positive")

        if self.search_window < 0:
            self.__log_and_exit(f"search_window must be positive")

        if self.fmin < 0:
            self.__log_and_exit(f"fmin must be positive")

        if self.fmax < 0:
            self.__log_and_exit(f"fmax must be positive")

        if self.fmax <= self.fmin:
            self.__log_and_exit(f"fmax > fmin not fulfilled")

        if self.slice_len < 0:
            self.__log_and_exit(f"slice_len must be positive")

        if self.transition_slice_ratio < 0:
            self.__log_and_exit(f"transition_slice_ratio must be positive")

        if self.mode == "online":
            if self.dtw != "oltw":
                self.__log_and_exit("For `online` mode only `oltw` dtw is accepted")
            if self.cqt not in ("nsgt", "librosa_pseudo", "librosa_hybrid"):
                self.__log_and_exit(
                    "For `online` mode only `nsgt`, `librosa_pseudo` or `librosa_hybrid` cqt is accepted"
                )

        if not path.isfile(self.perf_wave_path):
            self.__log_and_exit(
                f"Performance WAVE file ({self.perf_wave_path}) does not exist"
            )

        if not path.isfile(self.score_midi_path):
            self.__log_and_exit(
                f"Score MIDI file ({self.score_midi_path}) does not exist"
            )

        if self.slice_len % 100 != 0 and self.mode == "offline" and self.cqt == "nsgt":
            self.__log_and_exit(
                f"slice_len ({self.slice_len}) for offline nsgt must be a multiple of 100"
            )

        if self.backend not in ("alignment", "timestamp"):
            self.__log_and_exit("backend must be one of `alignment` or `timestamp`")

        if self.play_performance_audio and not self.simulate_performance:
            self.__log_and_exit(
                "`simulate_performance` must be set to True to enable `play_performance_audio`"
            )

        # ---- MUTATIVE ----

        # quantize fmin and fmax
        fmin = quantise_hz_midi(self.fmin)
        self.__log(f"fmin quantised from {self.fmin} to {fmin}")
        self.fmin = fmin

        fmax = quantise_hz_midi(self.fmax)
        self.__log(f"fmax quantised from {self.fmax} to {fmax}")
        self.fmax = fmax

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")

    def __str__(self) -> str:
        self_dict = self.as_dict()
        del self_dict["sanitize"]
        return "\n".join([f"--{arg} {val}" for (arg, val) in self_dict.items()])
