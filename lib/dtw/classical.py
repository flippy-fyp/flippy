from lib.eprint import eprint
from typing import List
from lib.dtw.shared import cost
from lib.sharedtypes import DTWPathElemType, ExtractedFeature
import numpy as np


class ClassicalDTW:
    def __init__(self, P: List[ExtractedFeature], S: List[ExtractedFeature]):
        if len(P) == 0:
            raise ValueError(f"Empty P")
        if len(S) == 0:
            raise ValueError(f"Empty S")
        if len(P[0].shape) != 1:
            raise ValueError(f"P must be 2D")
        if len(S[0].shape) != 1:
            raise ValueError(f"S must be 2D")

        import sys

        sys.setrecursionlimit(10 ** 7)
        self.__log("Set recursion limit to 10**7")

        self.P = P  # performance
        self.S = S  # score

        self.D_shape = (len(P), len(S))
        self.D_calc = np.zeros(
            self.D_shape, dtype=bool
        )  # whether the entry in self.D is calculated
        self.D = np.zeros(self.D_shape, dtype=np.float64)

        self.__log("Initialised successfully")

    def dtw(self) -> List[DTWPathElemType]:
        """
        Perform dtw.
        Returns the path (shape: (len(S), 2)) from (0, 0) to (len(S), len(P))
        """
        path: List[DTWPathElemType] = []

        r, c = self.D_shape
        r -= 1
        c -= 1
        self.__dtw_helper(r, c)

        while r >= 0 and c >= 0:
            path.append((r, c))
            r_pos = r > 0
            c_pos = c > 0
            diag_cost = np.inf
            left_cost = np.inf
            down_cost = np.inf

            if r_pos and c_pos:
                diag_cost = self.D[r - 1][c - 1]
            if r_pos:
                left_cost = self.D[r - 1][c]
            if c_pos:
                down_cost = self.D[r][c - 1]

            min_cost = min(diag_cost, left_cost, down_cost)

            if min_cost == diag_cost:
                # prefer diag if tie
                r -= 1
                c -= 1
            elif min_cost == left_cost:
                r -= 1
            else:
                c -= 1

        # Reverse path.
        path = list(reversed(path))

        return path

    def __dtw_helper(self, r: int, c: int) -> float:
        """
        Helper function. For an entry (r, c) in the distance matrix return the current cumulative cost
        and the immediate backward direction of the optimal path.
        """
        if r >= self.D_shape[0] or c >= self.D_shape[1]:
            raise ValueError(
                f"Out of range: want ({r}, {c}) from distance matrix of shape {self.D_shape}"
            )
        if r < 0 or c < 0:
            return np.inf

        if not self.D_calc[r][c]:
            self.D_calc[r][c] = True
            p = self.P[r]
            s = self.S[c]
            d = cost(s, p)

            if (r, c) == (0, 0):
                self.D[r][c] = d
            else:
                half_diag_cost = self.__dtw_helper(r - 1, c - 1)
                diag_cost = half_diag_cost * 2
                down_cost = self.__dtw_helper(r - 1, c)
                left_cost = self.__dtw_helper(r, c - 1)

                min_cost = min(diag_cost, down_cost, left_cost)
                curr_cost = d + min_cost
                self.D[r][c] = curr_cost

        return self.D[r][c]

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
