from typing import TypedDict


class NoteInfo(TypedDict):
    note_start: float  # note start time (ms)
    midi_note_num: int  # MIDI note number
