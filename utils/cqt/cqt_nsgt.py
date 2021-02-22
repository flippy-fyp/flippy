from utils.utils import quantise_hz_midi
from nsgt import CQ_NSGT_sliced, CQ_NSGT  # type: ignore
import librosa  # type: ignore
from typing import Tuple

def get_nsgt_params(
    fmin: float = 130.8,
    fmax: float = 4186.0,
) -> Tuple[float, float]:
    """
    Given fmin and fmax return (quantised fmin, quantised fmax)
    """
    return quantise_hz_midi(fmin), quantise_hz_midi(fmax)