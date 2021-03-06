from ..mputils import write_list_to_queue
from ..cqt.base import BaseCQT
from ..eprint import eprint
from ..sharedtypes import CQTType, ExtractedFeature, ExtractedFeatureQueue, ModeType
from ..cqt.cqt_nsgt import CQTNSGTSlicq, CQTNSGT
from typing import Callable, Optional, Dict, List, Union
from ..cqt.cqt_librosa import (
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
        self.__sleep_if_performance(self.frame_length, time.perf_counter() + 0.2)

        for s in audio_stream:
            pre_sleep_time = time.perf_counter()
            self.slice_queue.put(s)
            # sleep for hop length if performance
            self.__sleep_if_performance(self.hop_length, pre_sleep_time)

        self.slice_queue.put(None)  # end
        self.__log("Finished")

    def __sleep_if_performance(self, samples: int, pre_sleep_time: float):
        if self.simulate_performance:
            sleep_time = float(samples) / self.sample_rate
            time.sleep(sleep_time - (time.perf_counter() - pre_sleep_time) - 0.0005)

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
        hop_len: int,
        slice_hop_ratio: int,
        sample_rate: int,
        nsgt_multithreading: bool = False,
    ):
        self.mode = mode
        self.slice_queue = slice_queue
        self.output_queue = output_queue
        fmin, n_bins = get_librosa_params(fmin, fmax)

        extractor_map: Dict[ModeType, Dict[CQTType, Callable[[], BaseCQT]]] = {
            "offline": {
                "librosa": lambda: LibrosaFullCQT(
                    "librosa", fmin, n_bins, hop_len, sample_rate
                ),
                "librosa_hybrid": lambda: LibrosaFullCQT(
                    "librosa_hybrid", fmin, n_bins, hop_len, sample_rate
                ),
                "librosa_pseudo": lambda: LibrosaFullCQT(
                    "librosa_pseudo", fmin, n_bins, hop_len, sample_rate
                ),
                "nsgt": lambda: CQTNSGT(
                    hop_len * slice_hop_ratio,
                    slice_hop_ratio,
                    fmin,
                    fmax,
                    sample_rate,
                    nsgt_multithreading,
                ),
            },
            "online": {
                "librosa_hybrid": lambda: LibrosaSliceCQT(
                    "librosa_hybrid", fmin, n_bins, sample_rate
                ),
                "librosa_pseudo": lambda: LibrosaSliceCQT(
                    "librosa_pseudo", fmin, n_bins, sample_rate
                ),
                "nsgt": lambda: CQTNSGTSlicq(
                    hop_len * slice_hop_ratio,
                    slice_hop_ratio,
                    fmin,
                    fmax,
                    sample_rate,
                    nsgt_multithreading,
                ),
            },
        }
        mode_map = extractor_map.get(mode)
        if mode_map is None:
            raise ValueError(f"Unknown mode: {mode}")
        extractor_lambda = mode_map.get(cqt)
        if extractor_lambda is None:
            raise ValueError(
                f"Invalid or unknown combination of mode {mode} and cqt type {cqt}"
            )
        self.__extractor = extractor_lambda()

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        while True:
            sl: Optional[np.ndarray] = self.slice_queue.get()
            if sl is None:
                break
            o: Optional[
                Union[ExtractedFeature, List[ExtractedFeature]]
            ] = self.__extractor.extract(sl)
            if self.mode == "online":
                # o is of type ExtractedFeature
                self.output_queue.put(o)
            elif self.mode == "offline":
                # slice_queue has the whole audio piece
                # o is of type List[ExtractedFeature]
                write_list_to_queue(o, self.output_queue)
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
        hop_len: int,
        slice_hop_ratio: int,
        # slicer
        wave_path: str,
        simulate_performance: bool,
        # extractor
        mode: ModeType,
        cqt: CQTType,
        fmin: float,
        fmax: float,
        # output features
        output_queue: ExtractedFeatureQueue,
        nsgt_multithreading: bool = False,
    ):
        self.sample_rate = sample_rate
        self.wave_path = wave_path
        self.hop_len = hop_len
        self.slice_hop_ratio = slice_hop_ratio
        self.simulate_performance = simulate_performance

        self.mode = mode
        self.cqt = cqt
        self.fmin = fmin
        self.fmax = fmax

        self.output_queue = output_queue

        self.nsgt_multithreading = nsgt_multithreading

        self.__log("Initialised successfully")

    def __get_frame_len(self) -> int:
        if self.cqt == "nsgt":
            return self.slice_hop_ratio * self.hop_len
        return self.hop_len

    def start(self):
        self.__log("Starting...")
        slice_queue: ExtractedFeatureQueue = mp.Queue()

        online_slicer_proc: Optional[mp.Process] = None

        if self.mode == "online":
            slicer = Slicer(
                self.wave_path,
                self.hop_len,
                self.__get_frame_len(),
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
            self.hop_len,
            self.slice_hop_ratio,
            self.sample_rate,
            self.nsgt_multithreading,
        )
        feature_extractor_proc = mp.Process(target=feature_extractor.start)
        feature_extractor_proc.start()

        if online_slicer_proc:
            online_slicer_proc.join()
        feature_extractor_proc.join()
        self.__log("Finished")

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
