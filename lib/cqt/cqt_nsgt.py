# from lib.cqt.cqt_shared import postprocess_full_cqt
from lib.utils import quantise_hz_midi
from nsgt import CQ_NSGT_sliced, CQ_NSGT  # type: ignore
from typing import Tuple
import numpy as np  # type: ignore

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
    n_bins: int = 12,
    fs: int = 44100,
    multithreading: bool = False,
) -> np.ndarray: 
    nsgt = CQ_NSGT(fmin, fmax, n_bins, fs, audio.size, reducedform=2, multithreading=multithreading, matrixform=True)
    # Forward transform
    cqt = nsgt.forward(audio)
    # Convert to ndarray
    cqt = np.asarray(cqt)
    # Take abs val
    cqt = np.abs(cqt, dtype=np.float32)

    return cqt

def extract_features_nsgt_slicq(
    audio_slice: np.ndarray,
    fmin: float = 130.8,
    fmax: float = 4186.0,
    n_bins: int = 12,
) -> np.ndarray:
    """
    Extract features for an audio slice, which is 4x the transition (hop) length.
    """
    sl_len = len(audio_slice)
    tr_len = sl_len // 4
    slicq = CQ_NSGT_sliced(
        fmin,
        fmax,
        n_bins,
        sl_len,
        tr_len,
        matrixform=-True,
    )