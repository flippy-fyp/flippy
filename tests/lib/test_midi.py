import unittest
from os import path
from lib.midi import process_midi_to_note_info
from lib.sharedtypes import NoteInfo
from typing import List


class TestProcessMIDI(unittest.TestCase):
    def test_process_midi_to_note_info(self):
        midi_file_path = path.join(
            path.dirname(path.dirname(path.dirname(__file__))),
            "data",
            "sample_midis",
            "short_demo.mid",
        )
        want: List[NoteInfo] = [
            NoteInfo(60, 4.882802734375),
            NoteInfo(62, 514.6474082031249),
            NoteInfo(64, 1010.2518857421874),
            NoteInfo(64, 1505.8563632812497),
            NoteInfo(67, 1505.8563632812497),
        ]
        got = process_midi_to_note_info(midi_file_path)
        self.assertEqual(want, got)

    def test_process_midi_tracks(self):
        midi_file_path = path.join(
            path.dirname(path.dirname(path.dirname(__file__))),
            "data",
            "sample_midis",
            "short_demo_chord.mid",
        )
        want: List[NoteInfo] = [
            NoteInfo(60, 4.882802734375),
            NoteInfo(64, 4.882802734375),
        ]
        got = process_midi_to_note_info(midi_file_path)
        self.assertEqual(want, got)
