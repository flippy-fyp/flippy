import unittest
from flippy import add
from typing import List, Tuple


class TestAddMethod(unittest.TestCase):
    def test_add(self):
        cases: List[Tuple[Tuple[int, int], int]] = [
            ((3, 4), 7),
            ((-1, 1), 0),
        ]

        for tc in cases:
            inp, want = tc
            a, b = inp
            got = add(a, b)
            self.assertEqual(want, got)


if __name__ == "__main__":
    unittest.main()
