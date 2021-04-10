from typing import Callable, Literal, NewType, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import multiprocessing as mp


@dataclass
class NoteInfo:
    midi_note_num: int  # MIDI note number
    note_start: float  # note start time (ms)


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

ModeType = Literal["online", "offline"]
DTWType = Literal["classical", "oltw"]
CQTType = Literal["nsgt", "librosa_pseudo", "librosa_hybrid", "librosa"]
BackendType = Literal["alignment", "timestamp"]
