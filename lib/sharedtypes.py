from typing import Callable, Literal, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class NoteInfo:
    midi_note_num: int  # MIDI note number
    note_start: float  # note start time (ms)


ExtractorFunctionType = Callable[[np.ndarray], np.ndarray]

PIndex = int
SIndex = int

DTWPathElemType = Tuple[PIndex, SIndex]

ModeType = Literal["online", "offline"]
DTWType = Literal["classical", "oltw"]
CQTType = Literal["nsgt", "librosa_pseudo", "librosa_hybrid", "librosa"]
BackendType = Literal["alignment", "timestamp"]
