from lib.dtw.oltw import OLTW
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
                oltw = OLTW(mp.Queue(), S, mp.Queue(), 3, 250)
                oltw.dtw()
            self.assertTrue(excp_str in str(context.exception), excp_str)
