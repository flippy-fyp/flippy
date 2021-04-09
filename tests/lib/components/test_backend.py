from lib.components.backend import get_closest_note
from typing import Iterator, List, Optional, Tuple
from lib.sharedtypes import NoteInfo
from sortedcontainers import SortedDict  # type: ignore


from typing import List
import unittest


class TestGetClosestNote(unittest.TestCase):
    def test_with_expected(self):
        sorted_note_onsets: SortedDict[float, NoteInfo] = SortedDict(
            {
                100: NoteInfo(1, 100),
                200: NoteInfo(2, 200),
                300: NoteInfo(3, 300),
                400: NoteInfo(4, 400),
            }
        )
        testcases: List[Tuple(float, Optional[NoteInfo])] = [
            (0, NoteInfo(1, 100)),
            (100, NoteInfo(1, 100)),
            (150, NoteInfo(1, 100)),
            (151, NoteInfo(2, 200)),
        ]
        for (inp, want) in testcases:
            got = get_closest_note(sorted_note_onsets, inp)
            self.assertEqual(want, got)

    def test_with_none(self):
        sorted_note_onsets: SortedDict[float, NoteInfo] = SortedDict({})
        got = get_closest_note(sorted_note_onsets, 100)
        self.assertIsNone(got)
