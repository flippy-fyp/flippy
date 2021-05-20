from ..cqt.base import BaseCQT
from ..sharedtypes import ExtractedFeature
from ..utils import quantise_hz_midi
from nsgt import CQ_NSGT_sliced, CQ_NSGT  # type: ignore
from typing import List, Tuple
import librosa  # type: ignore
import numpy as np  # type: ignore
from ..constants import DEFAULT_SAMPLE_RATE


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
    tr_len: int = 2048,  # artificial
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

    # The "hop length" of CQ_NSGT is 100, so to simulate the provided hop_length, approximately split
    # and average the obtained results--this means that the time is approximate!
    averaged_cqt: List[np.ndarray] = []
    quantized_hop_length = tr_len // 100
    split_n = cqt.shape[0] // quantized_hop_length
    for i in range(split_n):
        start = i * quantized_hop_length
        end = start + quantized_hop_length
        avg = np.average(cqt[start:end], axis=0)

        averaged_cqt.append(avg)

    cqt = np.array(averaged_cqt)

    # Take abs value
    cqt = np.abs(cqt)
    # L1 normalize
    cqt = librosa.util.normalize(cqt, norm=1, axis=1)

    return cqt


def get_slicq_engine(
    sl_len: int,
    tr_len: int,
    fmin: float = 130.8,
    fmax: float = 4186.0,
    fs: int = DEFAULT_SAMPLE_RATE,
    multithreading: bool = False,
) -> CQ_NSGT_sliced:
    """
    Get slicq engine.
    """
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
    tr_len: int,
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
    cqt = cqt[:tr_len]
    # Average over the time slices
    cqt = np.average(cqt, axis=0)
    # L1 normalize
    cqt = librosa.util.normalize(cqt, norm=1)

    return cqt


class CQTNSGTSlicq(BaseCQT):
    def __init__(
        self,
        sl_len: int,
        tr_len: int,
        fmin: float = 130.8,
        fmax: float = 4186.0,
        fs: int = DEFAULT_SAMPLE_RATE,
        multithreading: bool = False,
    ):
        self.tr_len = tr_len
        self.slicq = get_slicq_engine(sl_len, tr_len, fmin, fmax, fs, multithreading)

    def extract(self, audio_slice: np.ndarray) -> ExtractedFeature:
        return extract_features_nsgt_slicq(self.slicq, self.tr_len, audio_slice)


class CQTNSGT(BaseCQT):
    def __init__(
        self,
        sl_len: int,
        tr_len: int,
        fmin: float = 130.8,
        fmax: float = 4186.0,
        fs: int = DEFAULT_SAMPLE_RATE,
        multithreading: bool = False,
    ):
        self.sl_len = sl_len
        self.tr_len = tr_len
        self.fmin = fmin
        self.fmax = fmax
        self.fs = fs
        self.multithreading = multithreading

    def extract(self, audio_slice: np.ndarray) -> List[ExtractedFeature]:
        slicq_extractor = CQTNSGTSlicq(
            self.sl_len,
            self.tr_len,
            self.fmin,
            self.fmax,
            self.fs,
            self.multithreading,
        )
        res: List[ExtractedFeature] = []
        hop_start: int = 0
        while hop_start <= len(audio_slice) - self.sl_len:
            feat = slicq_extractor.extract(
                audio_slice[hop_start : hop_start + self.sl_len]
            )
            res.append(feat)
            hop_start += self.tr_len
        return res

    def full_extract(self, audio_slice: np.ndarray) -> List[ExtractedFeature]:
        """
        The hop length is determined by fmin.
        """
        return [
            x
            for x in extract_features_nsgt_cqt(
                audio_slice,
                self.fmin,
                self.fmax,
                self.tr_len,
                self.fs,
                self.multithreading,
            )
        ]
