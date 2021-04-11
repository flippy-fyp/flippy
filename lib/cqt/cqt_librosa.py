from lib.cqt.base import BaseCQT
from typing import Any, Callable, Dict, Tuple
from lib.utils import quantise_hz_midi
import librosa  # type: ignore
import numpy as np  # type: ignore
from lib.sharedtypes import ExtractedFeature, ExtractorFunctionType, LibrosaCQTs
from lib.constants import DEFAULT_SAMPLE_RATE


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
    fs: int = DEFAULT_SAMPLE_RATE,
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
    fs: int = DEFAULT_SAMPLE_RATE,
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
    fs: int = DEFAULT_SAMPLE_RATE,
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
    fs: int = DEFAULT_SAMPLE_RATE,
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
    fs: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    cqt = cqt_func(
        audio_slice, sr=fs, hop_length=audio_slice.size, fmin=fmin, n_bins=n_bins
    )
    # Transpose so that each row is for a time slice's spectra
    cqt = cqt.T
    # Ignore the second element
    cqt = cqt[:1]
    # Reshape so that it is 1D
    cqt = cqt.reshape(cqt.shape[1])
    # Take abs value
    cqt = np.abs(cqt)
    # L1 normalize
    cqt = librosa.util.normalize(cqt, norm=1)

    return cqt


def extract_slice_features_librosa_pseudo_cqt(
    audio_slice: np.ndarray,
    fmin: float,
    n_bins: int,
    fs: int = DEFAULT_SAMPLE_RATE,
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
    fs: int = DEFAULT_SAMPLE_RATE,
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
#     fs: int = DEFAULT_SAMPLE_RATE,
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


class LibrosaSliceCQT(BaseCQT):
    def __init__(
        self, cqt: LibrosaCQTs, fmin: float, n_bins: int, fs: int = DEFAULT_SAMPLE_RATE
    ):
        self.fmin = fmin
        self.n_bins = n_bins
        self.fs = fs

        f_map: Dict[
            LibrosaCQTs, Callable[[np.ndarray, float, int, int], ExtractedFeature]
        ] = {
            "librosa_pseudo": extract_slice_features_librosa_pseudo_cqt,
            "librosa_hybrid": extract_slice_features_librosa_hybrid_cqt,
        }
        self.__f = f_map.get(cqt)
        if self.__f is None:
            raise ValueError(f"Unknown or unsupported cqt algo: {cqt}")

    def extract(self, audio_slice: np.ndarray) -> ExtractedFeature:
        return self.__f(audio_slice, self.fmin, self.n_bins, self.fs)  # type: ignore


class LibrosaFullCQT(BaseCQT):
    def __init__(
        self,
        cqt: LibrosaCQTs,
        fmin: float,
        n_bins: int,
        hop: int,
        fs: int = DEFAULT_SAMPLE_RATE,
    ):
        self.fmin = fmin
        self.n_bins = n_bins
        self.hop = hop
        self.fs = fs

        f_map: Dict[
            LibrosaCQTs, Callable[[np.ndarray, float, int, int, int], ExtractedFeature]
        ] = {
            "librosa": extract_features_librosa_cqt,
            "librosa_pseudo": extract_features_librosa_pseudo_cqt,
            "librosa_hybrid": extract_features_librosa_hybrid_cqt,
        }
        self.__f = f_map.get(cqt)
        if self.__f is None:
            raise ValueError(f"Unknown or unsupported cqt algo: {cqt}")

    def extract(self, audio_slice: np.ndarray) -> ExtractedFeature:
        return self.__f(audio_slice, self.fmin, self.n_bins, self.fs, self.hop)  # type: ignore
