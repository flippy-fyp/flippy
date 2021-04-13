from lib.cqt.base import BaseCQT
from lib.eprint import eprint
from lib.sharedtypes import CQTType, ExtractedFeatureQueue, LibrosaCQT, ModeType
from lib.cqt.cqt_nsgt import CQTNSGTSlicq, CQTNSGT
from typing import Callable, Optional, Dict
from lib.cqt.cqt_librosa import (
    LibrosaFullCQT,
    LibrosaSliceCQT,
    get_librosa_params,
)
import multiprocessing as mp
import librosa  # type: ignore
import time
import numpy as np


class Slicer:
    def __init__(
        self,
        wave_path: str,
        hop_length: int,
        frame_length: int,
        sample_rate: int,
        slice_queue: ExtractedFeatureQueue,
        simulate_performance: bool = True,
    ):
        self.wave_path = wave_path
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.slice_queue = slice_queue
        self.simulate_performance = simulate_performance
        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        audio_stream = librosa.stream(
            self.wave_path,
            1,
            self.frame_length,
            self.hop_length,
            mono=True,
            fill_value=0,
        )

        # before starting, sleep for frame_length if performance
        self.__sleep_if_performance(self.frame_length)

        for s in audio_stream:
            self.slice_queue.put(s)
            # sleep for hop length if performance
            self.__sleep_if_performance(self.hop_length)

        self.slice_queue.put(None)  # end
        self.__log("Finished")

    def __sleep_if_performance(self, samples: int):
        if self.simulate_performance:
            sleep_time = float(samples) / self.sample_rate
            time.sleep(sleep_time)

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")


class FeatureExtractor:
    def __init__(
        self,
        slice_queue: ExtractedFeatureQueue,
        output_queue: ExtractedFeatureQueue,
        mode: ModeType,
        cqt: CQTType,
        fmin: float,
        fmax: float,
        slice_len: int,
        transition_slice_ratio: int,
        sample_rate: int,
    ):
        self.mode = mode
        self.slice_queue = slice_queue
        self.output_queue = output_queue
        fmin, n_bins = get_librosa_params(fmin, fmax)

        extractor_map: Dict[ModeType, Dict[CQTType, BaseCQT]] = {
            "offline": {
                "librosa": LibrosaFullCQT(
                    "librosa", fmin, slice_len, n_bins, sample_rate
                ),
                "librosa_hybrid": LibrosaFullCQT(
                    "librosa_hybrid", fmin, slice_len, n_bins, sample_rate
                ),
                "librosa_pseudo": LibrosaFullCQT(
                    "librosa_pseudo", fmin, slice_len, n_bins, sample_rate
                ),
                "nsgt": CQTNSGT(fmin, fmax, slice_len, sample_rate),
            },
            "online": {
                "librosa_hybrid": LibrosaSliceCQT(
                    "librosa_hybrid", fmin, n_bins, sample_rate
                ),
                "librosa_pseudo": LibrosaSliceCQT(
                    "librosa_pseudo", fmin, n_bins, sample_rate
                ),
                "nsgt": CQTNSGTSlicq(
                    slice_len, transition_slice_ratio, fmin, fmax, sample_rate
                ),
            },
        }
        mode_map = extractor_map.get(mode)
        if mode_map is None:
            raise ValueError(f"Unknown mode: {mode}")
        self.__extractor = mode_map.get(cqt)
        if self.__extractor is None:
            raise ValueError(
                f"Invalid or unknown combination of mode {mode} and cqt type {cqt}"
            )

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        prev_slice: Optional[np.ndarray] = None
        while True:
            sl = self.slice_queue.get()
            if sl is None:
                break
            o = self.__extractor.extract(sl)
            if self.mode == "online":
                cqt_slice = (o - prev_slice).clip(0) if prev_slice is not None else o
                prev_slice = cqt_slice
                self.output_queue.put(cqt_slice)
            elif self.mode == "offline":
                # slice_queue has the whole audio piece and now we need to iterate and calculate the diffs
                for s in o:
                    cqt_slice = (
                        (s - prev_slice).clip(0) if prev_slice is not None else s
                    )
                    prev_slice = cqt_slice
                    self.output_queue.put(cqt_slice)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        self.output_queue.put(None)  # end
        self.__log("Finished")

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int,
        # slicer
        wave_path: str,
        hop_length: int,
        frame_length: int,
        simulate_performance: bool,
        # extractor
        mode: ModeType,
        cqt: CQTType,
        fmin: float,
        fmax: float,
        slice_len: int,
        transition_slice_ratio: int,
        # output features
        output_queue: ExtractedFeatureQueue,
    ):
        self.sample_rate = sample_rate
        self.wave_path = wave_path
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.simulate_performance = simulate_performance

        self.mode = mode
        self.cqt = cqt
        self.fmin = fmin
        self.fmax = fmax
        self.slice_len = slice_len
        self.transition_slice_ratio = transition_slice_ratio

        self.output_queue = output_queue

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        slice_queue: ExtractedFeatureQueue = mp.Queue()

        online_slicer_proc: Optional[mp.Process] = None

        if self.mode == "online":
            slicer = Slicer(
                self.wave_path,
                self.hop_length,
                self.frame_length,
                self.sample_rate,
                slice_queue,
                self.simulate_performance,
            )
            online_slicer_proc = mp.Process(target=slicer.start)
            online_slicer_proc.start()
        elif self.mode == "offline":
            audio, _ = librosa.load(self.wave_path, sr=self.sample_rate, mono=True)
            slice_queue.put(audio)
            slice_queue.put(None)  # end
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        feature_extractor = FeatureExtractor(
            slice_queue,
            self.output_queue,
            self.mode,
            self.cqt,
            self.fmin,
            self.fmax,
            self.slice_len,
            self.transition_slice_ratio,
            self.sample_rate,
        )
        feature_extractor_proc = mp.Process(target=feature_extractor.start)
        feature_extractor_proc.start()

        if online_slicer_proc:
            online_slicer_proc.join()

        feature_extractor_proc.join()
        self.__log("Finished")

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
