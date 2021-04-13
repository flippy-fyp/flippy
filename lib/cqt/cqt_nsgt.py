from lib.cqt.base import BaseCQT
from lib.sharedtypes import ExtractedFeature, ExtractorFunctionType
from lib.utils import quantise_hz_midi
from nsgt import CQ_NSGT_sliced, CQ_NSGT  # type: ignore
from typing import Tuple
import librosa  # type: ignore
import numpy as np  # type: ignore
from lib.constants import DEFAULT_SAMPLE_RATE


def get_nsgt_params(
    fmin: float = 130.8,
    fmax: float = 4186.0,
) -> Tuple[float, float]:
    """
    Given fmin and fmax return (quantised fmin, quantised fmax)
    """
    return quantise_hz_midi(fmin), quantise_hz_midi(fmax)


def extract_features_nsgt_cqt(
    audio: np.ndarray,
    fmin: float,
    fmax: float,
    hop_length: int = 2048,  # artificial
    fs: int = DEFAULT_SAMPLE_RATE,
    multithreading: bool = False,
) -> np.ndarray:
    nsgt = CQ_NSGT(
        fmin,
        fmax,
        12,
        fs,
        audio.size,
        reducedform=2,
        multithreading=multithreading,
        matrixform=True,
        real=True,
    )
    # Forward transform
    cqt = nsgt.forward(audio)
    # Convert to ndarray
    cqt = np.asarray(cqt)
    # Transpose so that each row is for a time slice's spectra
    cqt = cqt.T
    # Take abs value
    cqt = np.abs(cqt)

    # The "hop length" of CQ_NSGT is 100, so to simulate the provided hop_length, approximately split
    # and average the obtained results--this means that the time is approximate!
    averaged_cqt = np.empty((0, cqt.shape[1]), dtype=np.float64)
    quantized_hop_length = hop_length // 100
    split_n = cqt.shape[0] // quantized_hop_length
    for i in range(split_n):
        start = i * quantized_hop_length
        end = start + quantized_hop_length
        avg = np.average(cqt[start:end], axis=0)

        averaged_cqt = np.vstack([averaged_cqt, avg])

    cqt = averaged_cqt

    # L1 normalize
    cqt = librosa.util.normalize(cqt, norm=1, axis=1)
    # Take the first element to pad later
    cqt_0 = cqt[0]
    # Calculate diff between consecutive rows
    cqt = np.diff(cqt, axis=0)
    # Clip negatives
    cqt = cqt.clip(0)
    # Insert the first element
    cqt = np.insert(cqt, 0, cqt_0, axis=0)

    return cqt


class CQTNSGT(BaseCQT):
    def __init__(
        self,
        fmin: float,
        fmax: float,
        hop_length: int = 2048,  # artificial
        fs: int = DEFAULT_SAMPLE_RATE,
        multithreading: bool = False,
    ):
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = hop_length
        self.fs = fs
        self.multithreading = multithreading

    def extract(self, audio_slice: np.ndarray) -> ExtractedFeature:
        return extract_features_nsgt_cqt(
            audio_slice,
            self.fmin,
            self.fmax,
            self.hop_length,
            self.fs,
            self.multithreading,
        )


def get_slicq_engine(
    sl_len: int,
    sl_tr_ratio: int,
    fmin: float = 130.8,
    fmax: float = 4186.0,
    fs: int = DEFAULT_SAMPLE_RATE,
    multithreading: bool = False,
) -> CQ_NSGT_sliced:
    """
    Get slicq engine.
    """
    tr_len = sl_len // sl_tr_ratio
    return CQ_NSGT_sliced(
        fmin,
        fmax,
        12,
        sl_len,
        tr_len,
        fs,
        matrixform=True,
        reducedform=2,
        multithreading=multithreading,
        real=True,
    )


def extract_features_nsgt_slicq(
    slicq: CQ_NSGT_sliced,
    sl_tr_ratio: int,
    audio_slice: np.ndarray,
) -> np.ndarray:
    """
    Extract features for an audio slice.

    Based on https://github.com/sevagh/Music-Separation-TF/blob/master/algorithms/HPSS_CQNSGT_realtime.py
    """
    signal = (audio_slice,)
    # Forward transform
    cqt = slicq.forward(signal)
    # Convert to ndarray
    cqt = np.asarray(list(cqt))
    # Take abs value
    cqt = np.abs(cqt)
    # Average along the 3-element first axis
    cqt = np.average(cqt, axis=0)
    # Transpose so that each row is for a time slice's spectra
    cqt = cqt.T
    # Take only the slice region of the time slices
    cqt = cqt[: len(cqt) // sl_tr_ratio]
    # Average over the time slices
    cqt = np.average(cqt, axis=0)
    # L1 normalize
    cqt = librosa.util.normalize(cqt, norm=1)

    return cqt


class CQTNSGTSlicq(BaseCQT):
    def __init__(
        self,
        sl_len: int,
        sl_tr_ratio: int,
        fmin: float = 130.8,
        fmax: float = 4186.0,
        fs: int = DEFAULT_SAMPLE_RATE,
        multithreading: bool = False,
    ):
        self.slicq = get_slicq_engine(
            sl_len, sl_tr_ratio, fmin, fmax, fs, multithreading
        )
        self.sl_tr_ratio = sl_tr_ratio

    def extract(self, audio_slice: np.ndarray) -> ExtractedFeature:
        return extract_features_nsgt_slicq(self.slicq, self.sl_tr_ratio, audio_slice)
