from lib.eprint import eprint
from lib.midi import midi_to_audio
import tempfile
import os
import soundfile as sf  # type: ignore
from lib.constants import DEFAULT_SAMPLE_RATE


class Synthesiser:
    """
    Preprocess score
    """

    def __init__(self, score_midi_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.score_midi_path = score_midi_path
        self.sample_rate = sample_rate
        self.__log("Initialised successfully")

    def synthesise(self) -> str:
        """
        Returns path to synthesised wave file.
        """
        audio = midi_to_audio(self.score_midi_path)
        tmpdir = tempfile.mkdtemp(prefix="flippy")
        base_name_no_ext = os.path.splitext(os.path.basename(self.score_midi_path))[0]
        tmppath = os.path.join(tmpdir, f"{base_name_no_ext}.wav")

        sf.write(tmppath, audio, samplerate=self.sample_rate)

        return tmppath

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
