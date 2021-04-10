from lib.sharedtypes import CQTType, ModeType
from lib.cqt.cqt_nsgt import nsgt_extractor, nsgt_slicq_extractor
from typing import Optional
from lib.cqt.cqt_librosa import (
    get_extract_features_wrapper,
    get_extract_slice_features_wrapper,
    get_librosa_params,
)
import multiprocessing as mp
import numpy as np
import librosa  # type: ignore
import time


class Slicer:
    def __init__(
        self,
        wave_path: str,
        hop_length: int,
        frame_length: int,
        sample_rate: int,
        slice_queue: "mp.Queue[Optional[np.ndarray]]",
        simulate_performance: bool = True,
    ):
        self.wave_path = wave_path
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.slice_queue = slice_queue
        self.simulate_performance = simulate_performance

    def start(self):
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


class FeatureExtractor:
    def __init__(
        self,
        slice_queue: "mp.Queue[Optional[np.ndarray]]",
        output_queue: "mp.Queue[Optional[np.ndarray]]",
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
        if mode == "offline":
            if "librosa" in cqt:
                hop = slice_len
                self.extractor = get_extract_features_wrapper(cqt, fmin, hop, n_bins)
            elif "nsgt" == cqt:
                self.extractor = nsgt_extractor(fmin, fmax, slice_len)
            else:
                raise ValueError(f"Unknown cqt: {cqt}")
        elif mode == "online":
            if "librosa" in cqt:
                self.extractor = get_extract_slice_features_wrapper(cqt, fmin, n_bins)
            elif "nsgt" == cqt:
                self.extractor = nsgt_slicq_extractor(
                    slice_len, transition_slice_ratio, fmin, fmax
                )
            else:
                raise ValueError(f"Unknown cqt: {cqt}")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def start(self):
        while 1:
            sl = self.slice_queue.get()
            if sl is None:
                break
            o = self.extractor(sl)
            self.output_queue.put(o)
        self.output_queue.put(None)  # end


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
        output_queue: "mp.Queue[Optional[np.ndarray]]",
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

    def start(self):
        slice_queue: "mp.Queue[Optional[np.ndarray]]" = mp.Queue()

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
