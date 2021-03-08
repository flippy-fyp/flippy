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
from lib.constants import DEFAULT_SAMPLE_RATE


class Slicer:
    def __init__(
        self,
        wave_path: str,
        hop_length: int,
        frame_length: int,
        slice_queue: "mp.Queue[Optional[np.ndarray]]",
    ):
        self.wave_path = wave_path
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.slice_queue = slice_queue

    def start(self):
        audio_stream = librosa.stream(
            self.wave_path,
            1,
            self.frame_length,
            self.hop_length,
            mono=True,
            fill_value=0,
        )

        for s in audio_stream:
            self.slice_queue.put(s)
        self.slice_queue.put(None)  # end


class FeatureExtractor:
    def __init__(
        self,
        slice_queue: "mp.Queue[Optional[np.ndarray]]",
        output_queue: "mp.Queue[Optional[np.ndarray]]",
        mode: str,
        cqt: str,
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
        # slicer
        wave_path: str,
        hop_length: int,
        frame_length: int,
        # extractor
        mode: str,
        cqt: str,
        fmin: float,
        fmax: float,
        slice_len: int,
        transition_slice_ratio: int,
        # output features
        output_queue: "mp.Queue[Optional[np.ndarray]]",
    ):
        self.slice_queue: "mp.Queue[Optional[np.ndarray]]" = mp.Queue()

        online_slicer_proc: Optional[mp.Process] = None

        if mode == "online":
            slicer = Slicer(wave_path, hop_length, frame_length, self.slice_queue)
            online_slicer_proc = mp.Process(target=slicer.start)
            online_slicer_proc.start()
        elif mode == "offline":
            audio, _ = librosa.load(wave_path, sr=DEFAULT_SAMPLE_RATE, mono=True)
            self.slice_queue.put(audio)
            self.slice_queue.put(None)  # end
        else:
            raise ValueError(f"Unknown mode: {mode}")

        feature_extractor = FeatureExtractor(
            self.slice_queue,
            output_queue,
            mode,
            cqt,
            fmin,
            fmax,
            slice_len,
            transition_slice_ratio,
        )
        feature_extractor_proc = mp.Process(target=feature_extractor.start)
        feature_extractor_proc.start()

        if online_slicer_proc:
            online_slicer_proc.join()

        feature_extractor_proc.join()
