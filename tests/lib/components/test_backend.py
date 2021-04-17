from lib.components.backend import get_closest_notes_before
from typing import List, Optional, Tuple
from lib.sharedtypes import NoteInfo
from sortedcontainers import SortedDict  # type: ignore


from typing import List
import unittest


class TestGetClosestNotesBefore(unittest.TestCase):
    def test_with_expected(self):
        sorted_note_onsets: SortedDict[float, List[NoteInfo]] = SortedDict(
            {
                100: [NoteInfo(1, 100)],
                200: [NoteInfo(2, 200)],
                300: [NoteInfo(3, 300)],
                400: [NoteInfo(4, 400), NoteInfo(5, 400)],
            }
        )
        testcases: List[Tuple(float, Optional[List[NoteInfo]])] = [
            (0, []),
            (100, [NoteInfo(1, 100)]),
            (150, [NoteInfo(1, 100)]),
            (199, [NoteInfo(1, 100)]),
            (201, [NoteInfo(2, 200)]),
            (411, [NoteInfo(4, 400), NoteInfo(5, 400)]),
        ]
        for (inp, want) in testcases:
            got = get_closest_notes_before(sorted_note_onsets, inp)
            self.assertEqual(want, got)
