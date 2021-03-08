from lib.dtw.oltw import OLTW
from lib.sharedtypes import DTWPathType
from lib.mputils import consume_queue, produce_queue
from typing import List, Tuple
import unittest
import multiprocessing as mp
import numpy as np


class TestOLTW(unittest.TestCase):
    def test_oltw_constructor(self):
        # S, partial exception string
        testcases: List[Tuple[np.ndarray, str]] = [
            (
                np.empty((0, 0)),
                "Empty S",
            ),
            (
                np.empty(3),
                "S must be a 2D ndarray",
            ),
        ]

        for S, excp_str in testcases:
            with self.assertRaises(Exception, msg=excp_str) as context:
                oltw = OLTW(mp.Queue(), S, mp.Queue(), 999, 3)
                oltw.dtw()
            self.assertTrue(excp_str in str(context.exception), excp_str)

    def test_oltw(self):
        testcases: List[str, Tuple[np.ndarray, np.ndarray, DTWPathType]] = [
            (
                "Simple case",
                np.array([[1]], dtype=np.float32),
                np.array([[2]], dtype=np.float32),
                [(0, 0)],
            ),
            (
                "Report example",
                np.array([[1, 2], [3, 3], [2, 2], [2, 3], [6, 6]], dtype=np.float32),
                np.array([[1, 2], [3, 3], [2, 2], [4, 3], [2, 2]], dtype=np.float32),
                [(0, 0), (1, 1), (2, 2), (3, 2), (3, 3), (3, 4)],
            ),
        ]

        for name, P, S, want in testcases:
            output_queue = mp.Queue()
            P_queue = produce_queue(P)

            oltw = OLTW(P_queue, S, output_queue, 999, 3)
            oltw.dtw()

            got = consume_queue(output_queue)
            self.assertEqual(want, got, name)
