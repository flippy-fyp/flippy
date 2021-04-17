from pydub import AudioSegment  # type: ignore
import os
import tempfile


def cut_wave(start_s: float, end_s: float, file_path) -> str:
    """
    Cuts wave file in `file_path` from start_s to end_s.

    Saves and return the path to the cut wave file.
    """
    audio = AudioSegment.from_wav(file_path)
    audio = audio[int(start_s * 1000) : int(end_s * 1000)]

    tmpdir = tempfile.mkdtemp(prefix="flippy")
    base_name = os.path.basename(file_path)
    output_path = os.path.join(tmpdir, base_name)

    audio.export(output_path, format="wav")
    return output_path
