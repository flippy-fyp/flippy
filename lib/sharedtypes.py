from typing import Callable, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class NoteInfo:
    midi_note_num: int  # MIDI note number
    note_start: float  # note start time (ms)


ExtractorFunctionType = Callable[[np.ndarray], np.ndarray]

DTWPathElemType = Tuple[int, int]
