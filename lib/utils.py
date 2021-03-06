import librosa  # type: ignore

def quantise_hz_midi(hz: float) -> float:
    return librosa.midi_to_hz(round(librosa.hz_to_midi(hz)))
        