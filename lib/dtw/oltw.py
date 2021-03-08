from lib.dtw.shared import cost
from lib.sharedtypes import DTWPathElemType
import multiprocessing as mp
from typing import Optional, Set, Tuple
import numpy as np
from enum import Enum


class Direction(Enum):
    I = 1
    J = 2


DIR_I = set([Direction.I])
DIR_J = set([Direction.J])
DIR_IJ = set([Direction.I, Direction.J])


class OLTW:
    def __init__(
        self,
        P_queue: "mp.Queue[Optional[np.ndarray]]",
        S: np.ndarray,
        output_queue: "mp.Queue[Optional[DTWPathElemType]]",
        max_run_count: int,
        search_window: int,  # c
    ):
        if len(S.shape) != 2:
            raise ValueError(f"S must be a 2D ndarray, got shape: {S.shape}")
        if len(S) == 0:
            raise ValueError(f"Empty S")

        self.S = S  # score
        self.P_queue = P_queue
        self.output_queue = output_queue
        self.MAX_RUN_COUNT = max_run_count
        self.run_count: int = 1
        self.C = search_window
        self.D = np.ones((0, self.S.shape[0]), dtype=np.float32) * np.inf
        self.P = np.zeros((0, self.S.shape[1]), dtype=np.float32)

    def dtw(self):
        """
        Performs oltw
        Writes to self.output_queue
        """
        ### Step 1
        i = 0
        j = 0
        current: Set[Direction] = set()
        previous: Set[Direction] = set()

        i_prime = 0
        j_prime = 0
        # write to output queue
        self.output_queue.put((i_prime, j_prime))

        ### Step 2
        p_i = self.P_queue.get()
        if p_i is None:
            self.output_queue.put(None)
            return
        self.P = np.vstack([self.P, p_i])
        s_j = self.S[j]

        ### Step 3
        d = cost(p_i, s_j)
        self.__D_set(i, j, d)
        # print(np.flipud(self.D.T))

        while True:
            # Abort if last of S reached
            if j == len(self.S) - 1:
                self.output_queue.put(None)
                return

            ### Step 4
            current = self.__get_next_direction(i, j, i_prime, j_prime, previous)

            ### Step 5
            if Direction.I in current:
                # increment i
                i += 1
                # obtain p_i
                p_i = self.P_queue.get()
                if p_i is None:
                    self.output_queue.put(None)
                    return
                self.P = np.vstack([self.P, p_i])
                # compute required D elements
                for J in range(max(0, j - self.C + 1), j + 1):
                    s_J = self.S[J]
                    d = cost(p_i, s_J)
                    self.__D_set(i, J, d)

            if Direction.J in current:
                # increment j
                j += 1
                s_j = self.S[j]
                # compute required D elements
                for I in range(max(0, i - self.C + 1), i + 1):
                    p_I = self.P[I]
                    d = cost(p_I, s_j)
                    self.__D_set(I, j, d)

            ### Step 6
            if current == previous and previous != DIR_IJ:
                self.run_count += 1
            else:
                self.run_count = 1
            previous = current

            ### update i_prime and j_prime and write to output_queue
            i_prime, j_prime = self.__get_i_j_prime(i, j)
            # print(np.flipud(self.D.T))
            self.output_queue.put((i_prime, j_prime))

    def __get_i_j_prime(self, i: int, j: int) -> Tuple[int, int]:
        i_prime, j_prime = (i, j)
        min_D = np.inf
        curr_i = i
        while curr_i >= 0 and curr_i > (i - self.C):
            if self.D[curr_i][j] < min_D:
                min_D = self.D[curr_i][j]
                i_prime = curr_i
            curr_i -= 1
        curr_j = j
        while curr_j >= 0 and curr_j > (j - self.C):
            if self.D[i][curr_j] < min_D:
                min_D = self.D[i][curr_j]
                i_prime = i
                j_prime = curr_j
            curr_j -= 1
        return i_prime, j_prime

    def __get_next_direction(
        self, i: int, j: int, i_prime: int, j_prime: int, previous: Set[Direction]
    ) -> Set[Direction]:
        if i < self.C:
            return DIR_IJ
        elif self.run_count > self.MAX_RUN_COUNT:
            if previous == DIR_I:
                return DIR_J
            return DIR_I

        if i_prime < i:
            return DIR_J
        elif j_prime < j:
            return DIR_I
        return DIR_IJ

    def __D_set(self, i: int, j: int, d: np.float32):
        """
        at (i, j) and cost d assign to self.D
        """
        # print(i, j, d)
        while i >= self.D.shape[0]:
            self.D = np.vstack([self.D, np.ones(self.S.shape[0]) * np.inf])
        if (i, j) == (0, 0):
            self.D[i][j] = d
        else:
            self.D[i][j] = d + min(
                2 * self.__D_get(i - 1, j - 1),
                self.__D_get(i - 1, j),
                self.__D_get(i, j - 1),
            )

    def __D_get(self, i: int, j: int) -> np.float32:
        if i >= self.D.shape[0] or j >= self.D.shape[1]:
            raise ValueError(
                f"Out of range: want ({i}, {j}) from distance matrix of shape {self.D.shape}"
            )
        if i < 0 or j < 0:
            return np.float32(np.inf)

        return self.D[i][j]
