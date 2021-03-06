from typing import List
from .sharedtypes import NoteInfo
import pretty_midi  # type: ignore
from pretty_midi.pretty_midi import PrettyMIDI  # type: ignore
import numpy as np
import mido  # type: ignore
from itertools import chain
from .constants import DEFAULT_SAMPLE_RATE


def midi_to_audio(midi_path, fs=DEFAULT_SAMPLE_RATE) -> np.ndarray:
    midi_object: PrettyMIDI = pretty_midi.PrettyMIDI(midi_path)
    audio: np.ndarray = midi_object.fluidsynth(fs=fs)
    return audio


def process_midi_to_note_info(midi_path: str) -> List[NoteInfo]:
    """
    https://github.com/flippy-fyp/flippy-quantitative-testbench/blob/main/midi.py
    """
    mid = mido.MidiFile(midi_path)
    ret = process_MidiFile(mid)

    return ret


def process_MidiFile(mid: mido.MidiFile) -> List[NoteInfo]:
    tempo = get_tempo(mid.tracks[0])
    track_midi_note_info_ticks: List[List[NoteInfo]] = [
        process_track(track, mid.ticks_per_beat, tempo) for track in mid.tracks
    ]
    # flatten
    ret: List[NoteInfo] = list(chain.from_iterable(track_midi_note_info_ticks))
    # sort
    ret.sort(key=lambda x: x.note_start)
    return ret


def get_tempo(meta_track: mido.midifiles.tracks.MidiTrack) -> int:
    for msg in list(meta_track):
        if hasattr(msg, "tempo"):
            return msg.tempo
    raise ValueError("Cannot get track tempo")


def process_track(
    track: mido.midifiles.tracks.MidiTrack, ticks_per_beat: int, tempo: int
) -> List[NoteInfo]:
    ret: List[NoteInfo] = []
    curr_tick = 0
    for msg in track:
        curr_tick += msg.time
        if hasattr(msg, "velocity"):
            if msg.velocity > 0 and msg.type == "note_on":
                ret.append(
                    NoteInfo(
                        msg.note,
                        mido.tick2second(curr_tick, ticks_per_beat, tempo) * 1000,
                    )
                )
    return ret
