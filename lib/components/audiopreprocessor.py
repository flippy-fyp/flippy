from lib.eprint import eprint
from lib.sharedtypes import CQTType, ExtractedFeatureQueue, ModeType, ExtractedFeature
from lib.cqt.cqt_nsgt import nsgt_extractor, nsgt_slicq_extractor
from typing import Callable, Optional, Dict
from lib.cqt.cqt_librosa import (
    get_extract_features_wrapper,
    get_extract_slice_features_wrapper,
    get_librosa_params,
)
import multiprocessing as mp
import librosa  # type: ignore
import time


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
    ):
        self.slice_queue = slice_queue
        self.output_queue = output_queue
        fmin, n_bins = get_librosa_params(fmin, fmax)

        extractor_map: Dict[
            ModeType, Dict[CQTType, Callable[[ExtractedFeature], ExtractedFeature]]
        ] = {
            "offline": {
                "librosa": get_extract_features_wrapper(cqt, fmin, slice_len, n_bins),
                "librosa_hybrid": get_extract_features_wrapper(
                    cqt, fmin, slice_len, n_bins
                ),
                "librosa_pseudo": get_extract_features_wrapper(
                    cqt, fmin, slice_len, n_bins
                ),
                "nsgt": nsgt_extractor(fmin, fmax, slice_len),
            },
            "online": {
                "librosa": get_extract_slice_features_wrapper(cqt, fmin, n_bins),
                "librosa_hybrid": get_extract_slice_features_wrapper(cqt, fmin, n_bins),
                "librosa_pseudo": get_extract_slice_features_wrapper(cqt, fmin, n_bins),
                "nsgt": nsgt_slicq_extractor(
                    slice_len, transition_slice_ratio, fmin, fmax
                ),
            },
        }
        mode_map = extractor_map.get(mode)
        if mode_map is None:
            raise ValueError(f"Unknown mode: {mode}")
        self.__extractor = mode_map.get(cqt)
        if self.__extractor is None:
            raise ValueError(f"Unknown CQT type: {cqt}")

        self.__log("Initialised successfully")

    def start(self):
        self.__log("Starting...")
        while 1:
            sl = self.slice_queue.get()
            if sl is None:
                break
            o = self.__extractor(sl)
            self.output_queue.put(o)
        self.output_queue.put(None)  # end

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
                self.slice_queue,
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
        )
        feature_extractor_proc = mp.Process(target=feature_extractor.start)
        feature_extractor_proc.start()

        if online_slicer_proc:
            online_slicer_proc.join()

        feature_extractor_proc.join()

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
