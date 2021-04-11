from lib.eprint import eprint
from pydub import AudioSegment  # type: ignore
from pydub.playback import play  # type: ignore


class Player:
    """
    Plays wave file
    """

    def __init__(self, wave_file_path: str):
        self.wave = AudioSegment.from_wav(wave_file_path)
        self.__log("Initialised successfully")

    def play(self):
        play(self.wave)

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
