from lib.utils import quantise_hz_midi
from nsgt import CQ_NSGT_sliced, CQ_NSGT  # type: ignore
from typing import Tuple
import librosa  # type: ignore
import numpy as np  # type: ignore
import sys


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
    hop_length: int = 2048, # artificial
    fs: int = 44100,
    multithreading: bool = True,
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
    averaged_cqt = np.empty((0, cqt.shape[1]), dtype=np.float32)
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