from lib.sharedtypes import DTWPathElemType, ExtractedFeature
from lib.dtw.classical import ClassicalDTW
from typing import List, Tuple
import numpy as np
import unittest


class TestClassicalDTW(unittest.TestCase):
    def test_classical_exception(self):
        # P, S, partial exception str
        testcases: List[Tuple[List[ExtractedFeature], List[ExtractedFeature], str]] = [
            (
                [],
                [np.array([1, 2, 3], dtype=np.float64)],
                "Empty P",
            ),
            (
                [np.array([1, 2, 3], dtype=np.float64)],
                [],
                "Empty S",
            ),
            (
                [np.array([[1, 2, 3]], dtype=np.float64)],
                [np.array([1, 2, 3], dtype=np.float64)],
                "P must be 2D",
            ),
            (
                [np.array([1, 2, 3], dtype=np.float64)],
                [np.array([[1, 2, 3]], dtype=np.float64)],
                "S must be 2D",
            ),
        ]
        for P, S, excp_str in testcases:
            with self.assertRaises(Exception, msg=excp_str) as context:
                cdtw = ClassicalDTW(P, S, 1.0, 1.0, 1.0)
                cdtw.dtw()
            self.assertTrue(excp_str in str(context.exception), excp_str)

    def test_classical(self):
        testcases: List[
            str,
            Tuple[
                List[ExtractedFeature], List[ExtractedFeature], List[DTWPathElemType]
            ],
        ] = [
            (
                "Simple case",
                [np.array([1], dtype=np.float64)],
                [np.array([2], dtype=np.float64)],
                [(0, 0)],
            ),
            (
                "Report example",
                [
                    np.array([1, 2], dtype=np.float64),
                    np.array([3, 3], dtype=np.float64),
                    np.array([2, 2], dtype=np.float64),
                    np.array([2, 3], dtype=np.float64),
                    np.array([6, 6], dtype=np.float64),
                ],
                [
                    np.array([1, 2], dtype=np.float64),
                    np.array([3, 3], dtype=np.float64),
                    np.array([2, 2], dtype=np.float64),
                    np.array([4, 3], dtype=np.float64),
                    np.array([2, 2], dtype=np.float64),
                ],
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            ),
        ]

        for name, P, S, want in testcases:
            cdtw = ClassicalDTW(P, S, 1.0, 1.0, 1.0)
            got = cdtw.dtw()
            self.assertEqual(want, got, name)
