import pretty_midi
from pretty_midi.pretty_midi import PrettyMIDI
import numpy as np

def midi_to_audio(midi_path, fs=44100) -> np.ndarray:
    midi_object: PrettyMIDI = pretty_midi.PrettyMIDI(midi_path)
    audio: np.ndarray = midi_object.fluidsynth(fs=fs)
    return audio