from typing import Dict, Tuple
import numpy as np


class ClassicalDTW:
    def __init__(self, P: np.ndarray, S: np.ndarray):
        if len(P.shape) != 2:
            raise ValueError(f"P must be a 2D ndarray, got shape: {P.shape}")
        if len(S.shape) != 2:
            raise ValueError(f"S must be a 2D ndarray, got shape: {S.shape}")
        if len(P) == 0:
            raise ValueError(f"Empty P")
        if len(S) == 0:
            raise ValueError(f"Empty S")

        self.S = S  # score (rows)
        self.P = P  # performance (column)

        self.D_shape = (S.shape[0], P.shape[0])
        self.D_calc = np.zeros(
            self.D_shape, dtype=bool
        )  # whether the entry in self.D is calculated
        self.D = np.zeros(self.D_shape, dtype=np.float32)

        pass

    def dtw(self) -> np.ndarray:
        """
        Perform dtw.
        Returns the path (shape: (len(S), 2)) from (0, 0) to (len(S), len(P))
        """
        path = np.empty((0, 2), dtype=int)

        r, c = self.D_shape
        r -= 1
        c -= 1
        self._dtw_helper(r, c)

        while r >= 0 and c >= 0:
            path = np.vstack([path, [r, c]])
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

        # Flip array vertically.
        path = np.flipud(path)

        return path

    def _dtw_helper(self, r: int, c: int) -> float:
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
            s = self.S[r]
            p = self.P[c]
            d = np.sum(np.abs(s - p))

            if (r, c) == (0, 0):
                self.D[r][c] = d
            else:
                half_diag_cost = self._dtw_helper(r - 1, c - 1)
                diag_cost = half_diag_cost * 2
                down_cost = self._dtw_helper(r - 1, c)
                left_cost = self._dtw_helper(r, c - 1)

                min_cost = min(diag_cost, down_cost, left_cost)
                curr_cost = d + min_cost
                self.D[r][c] = curr_cost

        return self.D[r][c]
