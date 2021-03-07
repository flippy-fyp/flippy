import sys
from typing import Any, Callable, Tuple

from librosa.core import audio
from lib.utils import quantise_hz_midi
import librosa  # type: ignore
import numpy as np  # type: ignore


def get_librosa_params(
    fmin: float = 130.8,
    fmax: float = 4186.0,
) -> Tuple[float, int]:
    """
    Given fmin and fmax return (quantised fmin, n_bins)
    """
    # get n_bins
    start_midi = round(librosa.hz_to_midi(fmin))
    end_midi = round(librosa.hz_to_midi(fmax))
    n_bins = end_midi - start_midi

    # Round fmin to closest note's frequency
    fmin = quantise_hz_midi(fmin)

    return fmin, n_bins


def full_cqt_helper(
    cqt_func: Any,  # librosa.cqt, librosa.pseudo_cqt or librosa.hybrid_cqt
    audio: np.ndarray,
    fmin: float,
    n_bins: int,
    fs: int = 44100,
    hop: int = 2048,
) -> np.ndarray:

    # Compute CQT
    cqt = cqt_func(audio, sr=fs, hop_length=hop, fmin=fmin, n_bins=n_bins)
    # Transpose so that each row is for a time slice's spectra
    cqt = cqt.T
    # Take abs value
    cqt = np.abs(cqt)
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


def extract_features_librosa_cqt(
    audio: np.ndarray,
    fmin: float,
    n_bins: int,
    fs: int = 44100,
    hop: int = 2048,
) -> np.ndarray:
    """
    Extract features via librosa CQT.
    """
    return full_cqt_helper(
        librosa.cqt,
        audio,
        fmin,
        n_bins,
        fs=fs,
        hop=hop,
    )


def extract_features_librosa_pseudo_cqt(
    audio: np.ndarray,
    fmin: float,
    n_bins: int,
    fs: int = 44100,
    hop: int = 2048,
) -> np.ndarray:
    """
    Extract features via librosa pseudo CQT.
    """
    return full_cqt_helper(
        librosa.pseudo_cqt,
        audio,
        fmin,
        n_bins,
        fs=fs,
        hop=hop,
    )


def extract_features_librosa_hybrid_cqt(
    audio: np.ndarray,
    fmin: float,
    n_bins: int,
    fs: int = 44100,
    hop: int = 2048,
) -> np.ndarray:
    """
    Extract features via librosa hybrid CQT.
    """
    return full_cqt_helper(
        librosa.hybrid_cqt,
        audio,
        fmin,
        n_bins,
        fs=fs,
        hop=hop,
    )


def slice_cqt_helper(
    cqt_func: Any,  # librosa.cqt, librosa.pseudo_cqt or librosa.hybrid_cqt
    audio_slice: np.ndarray,
    fmin: float,
    n_bins: int,
    fs: int = 44100,
) -> np.ndarray:
    cqt = cqt_func(
        audio_slice, sr=fs, hop_length=audio_slice.size, fmin=fmin, n_bins=n_bins
    )
    # Transpose so that each row is for a time slice's spectra
    cqt = cqt.T
    # Ignore the second element
    cqt = cqt[:1]
    # Take abs value
    cqt = np.abs(cqt)
    # L1 normalize
    cqt = librosa.util.normalize(cqt, norm=1, axis=1)
    return cqt


def extract_slice_features_librosa_pseudo_cqt(
    audio_slice: np.ndarray,
    fmin: float,
    n_bins: int,
    fs: int = 44100,
) -> np.ndarray:
    """
    Extract features for an audio slice via librosa pseudo CQT.
    """
    return slice_cqt_helper(
        librosa.pseudo_cqt,
        audio_slice,
        fmin,
        n_bins,
        fs=fs,
    )


def extract_slice_features_librosa_hybrid_cqt(
    audio_slice: np.ndarray,
    fmin: float,
    n_bins: int,
    fs: int = 44100,
) -> np.ndarray:
    """
    Extract features for an audio slice via librosa hybrid CQT.
    """
    return slice_cqt_helper(
        librosa.hybrid_cqt,
        audio_slice,
        fmin,
        n_bins,
        fs=fs,
    )


## Experimental: Bad results
# def extract_slice_features_librosa_cqt(
#     audio_slice: np.ndarray,
#     fmin: float,
#     n_bins: int,
#     fs: int = 44100,
# ) -> np.ndarray:
#     """
#     Extract features for an audio slice via librosa CQT.
#     """
#     return slice_cqt_helper(
#         librosa.cqt,
#         audio_slice,
#         fmin,
#         n_bins,
#         fs=fs,
#     )


def get_extract_slice_features_wrapper(
    cqt: str,
    fmin: float,
    n_bins: int,
    fs: int = 44100,
) -> ExtractorFunctionType:
    f_map = {
        "librosa_pseudo": extract_slice_features_librosa_pseudo_cqt,
        "librosa_hybrid": extract_slice_features_librosa_hybrid_cqt,
    }
    if cqt not in f_map:
        raise ValueError(f"Unknown cqt algo: {cqt}")

    cqt_f = f_map[cqt]

    def w(audio_slice: np.ndarray) -> np.ndarray:
        return cqt_f(audio_slice, fmin, n_bins, fs)

    return w


def get_extract_features_wrapper(
    cqt: str,
    fmin: float,
    n_bins: int,
    hop: int,
    fs: int = 44100,
) -> ExtractorFunctionType:
    f_map = {
        "librosa": extract_features_librosa_cqt,
        "librosa_pseudo": extract_features_librosa_pseudo_cqt,
        "librosa_hybrid": extract_features_librosa_hybrid_cqt,
    }
    if cqt not in f_map:
        raise ValueError(f"Unknown cqt algo: {cqt}")

    cqt_f = f_map[cqt]

    def w(audio: np.ndarray) -> np.ndarray:
        return cqt_f(audio, fmin, n_bins, fs, hop)

    return w