from typing import Callable, TypedDict
import numpy as np


class NoteInfo(TypedDict):
    note_start: float  # note start time (ms)
    midi_note_num: int  # MIDI note number


ExtractorFunctionType = Callable[[np.ndarray], np.ndarray]
