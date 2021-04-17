from ..eprint import eprint
from ..midi import midi_to_audio
import tempfile
import os
import soundfile as sf  # type: ignore
from ..constants import DEFAULT_SAMPLE_RATE
from typing import Optional


class Synthesiser:
    """
    Preprocess score
    """

    def __init__(self, score_midi_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.score_midi_path = score_midi_path
        self.sample_rate = sample_rate
        self.__log("Initialised successfully")

    def synthesise(self, duration: Optional[float] = None) -> str:
        """
        Returns path to synthesised wave file.

        duration is the length of audio in seconds to write. None denotes write whole file.
        """
        tmpdir = tempfile.mkdtemp(prefix="flippy")
        base_name_no_ext = os.path.splitext(os.path.basename(self.score_midi_path))[0]
        tmppath = os.path.join(tmpdir, f"{base_name_no_ext}.wav")

        audio = midi_to_audio(self.score_midi_path, self.sample_rate)
        if duration is not None:
            full_duration = len(audio) / self.sample_rate
            if full_duration < duration:
                raise ValueError(
                    f"Duration ({duration}s) longer than whole audio's ({full_duration}s)"
                )
            if duration < 0:
                raise ValueError(f"Cannot take negative duration ({duration}s)")
            audio = audio[: int(duration * self.sample_rate)]
        sf.write(tmppath, audio, samplerate=self.sample_rate)

        return tmppath

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
