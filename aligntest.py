from typing import TypedDict, Dict, Set, Optional, List, Union, Tuple
import copy


class GElem:
    __allowed = ("c", "psi", "score", "left", "down", "diag")

    def __init__(self, **kwargs):
        self.c: Optional[str] = None
        self.psi: Optional[Set[str]] = None
        self.score: Optional[int] = None
        self.left = False
        self.down = False
        self.diag = False

        for k, v in kwargs.items():
            assert k in self.__class__.__allowed
            setattr(self, k, v)


Alignment = List[Tuple[str, Set[str]]]


class ASM:
    def __init__(self, P: List[str], Psi: List[Set[str]], simple=False):
        self.ALPHA = 1
        self.GAMMA = -1
        self.BETA_HAT = -10
        self.P = P
        self.Psi = Psi
        self.G: Dict[int, Dict[int, GElem]] = {}
        self.set_up()
        self.simple = simple

        self.solve()

        self.alignments: List[Alignment] = []

    def get_alignments(self) -> List[Alignment]:
        if not self.alignments:
            self.get_alignments_helper(1 + len(self.P), 1 + len(self.Psi), [])
        return self.alignments

    def get_alignments_helper(self, x: int, y: int, alignment: Alignment):
        if (x, y) == (1, 1):
            self.alignments.append(alignment)
            return

        g = self.get_G(x, y)

        psi = self.get_G(0, y).psi
        if psi is None:
            raise ValueError(f"Invalid y {y}")

        c = self.get_G(x, 0).c
        if c is None:
            raise ValueError(f"Invalid x {x}")

        if g.left:
            a = [(c, {'-'})] + copy.deepcopy(alignment)
            self.get_alignments_helper(x - 1, y, a)
        if g.down:
            a = [('-', psi)] + copy.deepcopy(alignment)
            self.get_alignments_helper(x, y - 1, a)
        if g.diag:
            a = [(c, psi)] + copy.deepcopy(alignment)
            self.get_alignments_helper(x - 1, y - 1, a)

    def solve(self):
        x = 1 + len(self.P)
        y = 1 + len(self.Psi)

        self.score(x, y)

    def __repr__(self) -> str:
        res = ""
        for y in range(1 + len(self.Psi), -1, -1):
            for x in range(0, 1 + len(self.P)):
                if (x, y) == (0, 0) or (x, y) == (1, 0) or (x, y) == (0, 1):
                    pass
                elif x == 0:
                    g = self.get_G(x, y)
                    psi = g.psi
                    if psi is None:
                        raise ValueError(f"psi is None in ({x}, {y})")
                    res += psi.__repr__()
                elif y == 0:
                    g = self.get_G(x, y)
                    c = g.c
                    if c is None:
                        raise ValueError(f"c is None in ({x}, {y})")
                    res += c
                else:
                    g = self.get_G(x, y)
                    sc = g.score
                    if sc is None:
                        raise ValueError(f"score is None in ({x}, {y})")
                    res += str(sc)
                    if g.left:
                        res += "l"
                    if g.diag:
                        res += "s"
                    if g.down:
                        res += "d"
                res += " & "
            res += "\\\\\n"
        return res

    def set_up(self):
        P = self.P
        Psi = self.Psi

        x: int = 2
        for c in P:
            self.set_G(x, 0, GElem(c=c))
            x += 1

        y: int = 2
        for psi in Psi:
            self.set_G(0, y, GElem(psi=psi))
            y += 1

        self.set_G(0, 0, GElem())
        self.set_G(1, 0, GElem())
        self.set_G(0, 1, GElem())
        self.set_G(1, 1, GElem(score=0))

        score_ctr = -1
        for x in range(2, 2 + len(P)):
            self.set_G(x, 1, GElem(score=score_ctr, left=True))
            score_ctr -= 1

        score_ctr = -1
        for y in range(2, 2 + len(Psi)):
            self.set_G(1, y, GElem(score=score_ctr, down=True))
            score_ctr -= 1

    def set_G(self, x: int, y: int, e: GElem):
        if x not in self.G:
            self.G[x] = {}
        self.G[x][y] = e

    def get_G(self, x: int, y: int) -> GElem:
        if x not in self.G:
            raise ValueError(f"No ({x}, ...) in G")
        if y not in self.G[x]:
            raise ValueError(f"No ({x}, {y}) in G")

        return self.G[x][y]

    def min_abs_dist(self, c: str, psi: Set[str]) -> int:
        res = 100000000
        for x in psi:
            res = min(res, abs(ord(c) - ord(x)))
        return res

    def beta(self, c: str, psi: Set[str]) -> int:
        return max(self.min_abs_dist(c, psi), self.BETA_HAT)

    def sim(self, c: str, psi: Set[str]) -> int:
        if c in psi:
            return self.ALPHA
        return self.beta(c, psi)

    def score(self, x, y) -> int:
        if x == 0 or y == 0:
            return -100000000

        sc = self.get_G(x, y).score if x in self.G and y in self.G[x] else None
        if sc is None:
            c = self.get_G(x, 0).c
            if c is None:
                raise ValueError(f"score called with invalid x: {x}")
            psi = self.get_G(0, y).psi
            if psi is None:
                raise ValueError(f"score called with invalid y: {y}")

            sim_val = 1 if c in psi else -1
            g = GElem()

            if self.simple:
                score_case_1 = self.score(x - 1, y - 1) + sim_val
                score_case_2 = self.score(x - 1, y) + self.GAMMA
                score_case_3 = self.score(x, y - 1) + self.GAMMA
            else:
                score_case_1 = self.score(x - 1, y - 1) + sim_val
                score_case_2 = self.score(x - 1, y) + (
                    min(0, sim_val) if x != 1 and y != 1 else self.GAMMA
                )
                score_case_3 = self.score(x, y - 1) + self.GAMMA

            sc = max(score_case_1, score_case_2, score_case_3)

            g.score = sc
            if score_case_1 == sc:
                g.diag = True
            if score_case_2 == sc:
                g.left = True
            if score_case_3 == sc:
                g.down = True

            self.set_G(x, y, g)
        return sc


if __name__ == "__main__":
    # P = ['A', 'C', 'B', 'A', 'C', 'A', 'D']
    # Psi: List[Set[str]] = [{'A'}, {'A', 'B', 'C'}, {'A', 'C'}, {'D'}]

    P = ["A", "B", "C", "D", "A", "B", "E"]
    Psi = [{"A"}, {"C"}, {"D"}, {"D"}, {"C"}, {"B"}, {"C"}]
    asm = ASM(P, Psi)
    print(asm)
    print("=====")
    alignments = asm.get_alignments()

    for a in alignments:
        print(a)