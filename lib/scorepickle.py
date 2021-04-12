from typing import List
from lib.sharedtypes import ExtractedFeature, NoteInfo
import tempfile
import os
import pickle


class ScorePickle:
    def __init__(self, note_onsets: List[NoteInfo], S: List[ExtractedFeature]):
        self.note_onsets = note_onsets
        self.S = S

    def dump(self, score_midi_path: str) -> str:
        tmpdir = tempfile.mkdtemp(prefix="flippy")
        base_name_no_ext = os.path.splitext(os.path.basename(score_midi_path))[0]
        tmppath = os.path.join(tmpdir, f"{base_name_no_ext}.pickle")

        with open(tmppath, "wb") as f:
            pickle.dump(self, f, protocol=4)

        return tmppath

    @staticmethod
    def load(file_path: str):
        with open(file_path, "rb") as f:
            score_pickle: ScorePickle = pickle.load(f)
            return score_pickle
