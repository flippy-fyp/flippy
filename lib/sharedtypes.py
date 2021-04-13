from typing import Callable, Literal, NewType, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import multiprocessing as mp


@dataclass
class NoteInfo:
    midi_note_num: int  # MIDI note number
    note_start: float  # note start time (ms)

    def __eq__(self, other):
        if not isinstance(other, NoteInfo):
            return False

        return (
            self.midi_note_num == other.midi_note_num
            and self.note_start == other.note_start
        )

    def __hash__(self):
        return hash((self.midi_note_num, self.note_start))


PIndex = int
SIndex = int

DTWPathElemType = Tuple[PIndex, SIndex]
FollowerOutputQueue = NewType(
    "FollowerOutputQueue", "mp.Queue[Optional[DTWPathElemType]]"
)
MultiprocessingConnection = NewType(
    "MultiprocessingConnection", "mp.connection.Connection"
)
ExtractedFeature = np.ndarray
ExtractedFeatureQueue = NewType(
    "ExtractedFeatureQueue", "mp.Queue[Optional[ExtractedFeature]]"
)

ExtractorFunctionType = Callable[[ExtractedFeature], ExtractedFeature]

LibrosaCQT = Literal["librosa"]
LibrosaPseudoCQT = Literal["librosa_pseudo"]
LibrosaHybridCQT = Literal["librosa_hybrid"]
LibrosaCQTs = Literal[LibrosaCQT, LibrosaPseudoCQT, LibrosaHybridCQT]
NSGTCQT = Literal["nsgt"]

ModeType = Literal["online", "offline"]
DTWType = Literal["classical", "oltw"]
CQTType = Literal[
    "librosa", "librosa_pseudo", "librosa_hybrid", "nsgt"
]  # due to limitation of TAP
BackendType = Literal["alignment", "timestamp"]
