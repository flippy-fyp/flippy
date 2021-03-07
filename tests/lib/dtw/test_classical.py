from lib.dtw.classical import ClassicalDTW
from typing import List, Tuple
import numpy as np
import unittest


class TestClassicalDTW(unittest.TestCase):
    def test_classical_exception(self):
        # P, S, partial exception str
        testcases: List[Tuple[np.ndarray, np.ndarray, str]] = [
            (
                np.empty((0, 0)),
                np.empty((0, 0)),
                "Empty P",
            ),
            (
                np.empty((3, 3)),
                np.empty((0, 0)),
                "Empty S",
            ),
            (
                np.empty(3),
                np.empty((0, 0)),
                "P must be a 2D ndarray",
            ),
            (
                np.empty((3, 3)),
                np.empty(3),
                "S must be a 2D ndarray",
            ),
        ]
        for P, S, excp_str in testcases:
            with self.assertRaises(Exception, msg=excp_str) as context:
                cdtw = ClassicalDTW(P, S)
                cdtw.dtw()
            self.assertTrue(excp_str in str(context.exception), excp_str)
                        
    def test_classical(self):
        testcases: List[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = [
            (
                "Simple case",
                np.array([[1]]),
                np.array([[2]]),
                np.array([[0, 0]])
            ),
            (
                "Report example",
                np.array([
                    [1, 2],
                    [3, 3],
                    [2, 2],
                    [2, 3],
                    [6, 6]
                ]),
                np.array([
                    [1, 2],
                    [3, 3],
                    [2, 2],
                    [4, 3],
                    [2, 2]
                ]),
                np.array([
                    [0, 0],
                    [1, 1],
                    [2, 2],
                    [3, 3],
                    [4, 3],
                    [4, 4]
                ])
            )
        ]

        for name, P, S, want in testcases:
            cdtw = ClassicalDTW(P, S)
            got = cdtw.dtw()   
            np.testing.assert_array_equal(want, got, err_msg=name)