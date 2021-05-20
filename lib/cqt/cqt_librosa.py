from ..cqt.base import BaseCQT
from typing import Any, Callable, Dict, List, Tuple
from ..utils import quantise_hz_midi
import librosa  # type: ignore
import numpy as np  # type: ignore
from ..sharedtypes import ExtractedFeature, ExtractorFunctionType, LibrosaCQTs
from ..constants import DEFAULT_SAMPLE_RATE


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
    frame_len: int,
    hop_len: int,
    fmin: float,
    n_bins: int,
    fs: int = DEFAULT_SAMPLE_RATE,
    consider_frames: bool = False,
) -> np.ndarray:
    if consider_frames:
        cqt = np.ndarray(
            [
                slice_cqt_helper(
                    cqt_func,
                    audio[i : min(len(audio), i + frame_len)],
                    hop_len,
                    fmin,
                    n_bins,
                    fs,
                )
                for i in range(0, len(audio) - frame_len + 1, hop_len)
            ]
        )
        return cqt
    else:
        # Compute CQT
        cqt = cqt_func(audio, sr=fs, hop_length=hop_len, fmin=fmin, n_bins=n_bins)
        # Transpose so that each row is for a time slice's spectra
        cqt = cqt.T
        # Take abs value
        cqt = np.abs(cqt)

        return cqt


def extract_features_librosa_cqt(
    audio: np.ndarray,
    frame_len: int,
    hop_len: int,
    fmin: float,
    n_bins: int,
    fs: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract features via librosa CQT.
    """
    return full_cqt_helper(
        librosa.cqt,
        audio,
        frame_len,
        hop_len,
        fmin,
        n_bins,
        fs,
        False,
    )


def extract_features_librosa_pseudo_cqt(
    audio: np.ndarray,
    frame_len: int,
    hop_len: int,
    fmin: float,
    n_bins: int,
    fs: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract features via librosa pseudo CQT.
    """
    return full_cqt_helper(
        librosa.pseudo_cqt,
        audio,
        frame_len,
        hop_len,
        fmin,
        n_bins,
        fs,
        True,
    )


def extract_features_librosa_hybrid_cqt(
    audio: np.ndarray,
    frame_len: int,
    hop_len: int,
    fmin: float,
    n_bins: int,
    fs: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract features via librosa hybrid CQT.
    """
    return full_cqt_helper(
        librosa.hybrid_cqt,
        audio,
        frame_len,
        hop_len,
        fmin,
        n_bins,
        fs,
        True,
    )


def slice_cqt_helper(
    cqt_func: Any,  # librosa.cqt, librosa.pseudo_cqt or librosa.hybrid_cqt
    audio_slice: np.ndarray,
    hop_len: int,
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
    # Take only the hop region of the time slices
    cqt = cqt[:hop_len]
    # L1 normalize
    cqt = librosa.util.normalize(cqt, norm=1)

    return cqt


def extract_slice_features_librosa_pseudo_cqt(
    audio_slice: np.ndarray,
    hop_len: int,
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
        hop_len,
        fmin,
        n_bins,
        fs=fs,
    )


def extract_slice_features_librosa_hybrid_cqt(
    audio_slice: np.ndarray,
    hop_len: int,
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
        hop_len,
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
        self,
        cqt: LibrosaCQTs,
        hop_len: int,
        fmin: float,
        n_bins: int,
        fs: int = DEFAULT_SAMPLE_RATE,
    ):
        self.fmin = fmin
        self.n_bins = n_bins
        self.fs = fs
        self.hop_len = hop_len

        f_map: Dict[
            LibrosaCQTs, Callable[[np.ndarray, int, float, int, int], ExtractedFeature]
        ] = {
            "librosa_pseudo": extract_slice_features_librosa_pseudo_cqt,
            "librosa_hybrid": extract_slice_features_librosa_hybrid_cqt,
        }
        self.__f = f_map.get(cqt)
        if self.__f is None:
            raise ValueError(f"Unknown or unsupported cqt algo: {cqt}")

    def extract(self, audio: np.ndarray) -> ExtractedFeature:
        return self.__f(audio, self.hop_len, self.fmin, self.n_bins, self.fs)  # type: ignore


class LibrosaFullCQT(BaseCQT):
    def __init__(
        self,
        cqt: LibrosaCQTs,
        frame_len: int,
        hop_len: int,
        fmin: float,
        n_bins: int,
        fs: int = DEFAULT_SAMPLE_RATE,
    ):
        self.fmin = fmin
        self.n_bins = n_bins
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.fs = fs

        f_map: Dict[
            LibrosaCQTs,
            Callable[[np.ndarray, int, int, float, int, int], ExtractedFeature],
        ] = {
            "librosa": extract_features_librosa_cqt,
            "librosa_pseudo": extract_features_librosa_pseudo_cqt,
            "librosa_hybrid": extract_features_librosa_hybrid_cqt,
        }
        self.__f = f_map.get(cqt)
        if self.__f is None:
            raise ValueError(f"Unknown or unsupported cqt algo: {cqt}")

    def extract(self, audio: np.ndarray) -> List[ExtractedFeature]:
        return [x for x in self.__f(audio, self.frame_len, self.hop_len, self.fmin, self.n_bins, self.fs)]  # type: ignore
