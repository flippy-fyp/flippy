from typing import TypedDict, Dict, Set, Optional, List, Union
import copy

class Elem(TypedDict):
    c: Optional[str]
    psi: Optional[Set[str]]
    score: Optional[int]
    left: bool
    down: bool
    diag: bool
    
def get_def_elem() -> Elem:
    return {
        'c': None,
        'psi': None,
        'score': None,
        'left': False,
        'down': False,
        'diag': False,
    }

class ASM():
    def __init__(self, P: List[str], Psi: List[Set[str]], simple=False):
        self.ALPHA = 1
        self.GAMMA = -1
        self.BETA_HAT = -10
        self.P = P
        self.Psi = Psi
        self.G: Dict[int, Dict[int, Elem]] = {}
        self.set_up()

        if simple:
            self.solve_simple()
        else:
            self.solve()
        
    def solve(self):
        x = 1 + len(self.P)
        y = 1 + len(self.Psi)

        self.score(x, y)

    def solve_simple(self):
        x = 1 + len(self.P)
        y = 1 + len(self.Psi)

        self.score_simple(x, y)


    def __repr__(self) -> str:
        res = ""
        for y in range(1 + len(self.Psi), -1, -1):
            for x in range(0, 1 + len(self.P)):
                if (x, y) == (0, 0) or (x, y) == (1, 0) or (x, y) == (0, 1):
                    pass
                elif x == 0:
                    g = self.get_G(x, y)
                    psi = g['psi']
                    if psi is None:
                        raise ValueError(f'psi is None in ({x}, {y})')
                    res += psi.__repr__()
                elif y == 0:
                    g = self.get_G(x, y)
                    c = g['c']
                    if c is None:
                        raise ValueError(f'c is None in ({x}, {y})')
                    res += c
                else:
                    g = self.get_G(x, y)
                    sc = g['score']
                    if sc is None:
                        raise ValueError(f'score is None in ({x}, {y})')
                    res += str(sc)
                    if g['left']: res += 'l'
                    if g['diag']: res += 's'
                    if g['down']: res += 'd'
                res += ' & '
            res += '\\\\\n'
        return res
            

        
    def set_up(self):
        P = self.P
        Psi = self.Psi

        x: int = 2
        for c in P:
            g = get_def_elem()
            g['c'] = c
            self.set_G(x, 0, g)
            x += 1

        y: int = 2
        for psi in Psi:
            g = get_def_elem()
            g['psi'] = psi
            self.set_G(0, y, g)
            y += 1

        g = get_def_elem()
        g['score'] = 0
        self.set_G(1, 1, g)

        score_ctr = -1
        for x in range(2, 2 + len(P)):
            g = get_def_elem()
            g['score'] = score_ctr
            self.set_G(x, 1, g)
            score_ctr -= 1

        score_ctr = -1
        for y in range(2, 2 + len(Psi)):
            g = get_def_elem()
            g['score'] = score_ctr
            self.set_G(1, y, g)
            score_ctr -= 1
        
        
    def set_G(self, x: int, y: int, e: Elem):
        if x not in self.G:
            self.G[x] = {}
        self.G[x][y] = e

    def get_G(self, x: int, y: int) -> Elem:
        if x not in self.G:
            raise ValueError(f'No ({x}, ...) in G')
        if y not in self.G[x]:
            raise ValueError(f'No ({x}, {y}) in G')

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

    def score_simple(self, x, y) -> int:
        if x == 0 or y == 0: 
            return -100000000

        sc = self.get_G(x, y)['score'] if x in self.G and y in self.G[x] else None
        if sc is None:
            c = self.get_G(x, 0)['c']
            if c is None:
                raise ValueError(f'score called with invalid x: {x}')
            psi = self.get_G(0, y)['psi']
            if psi is None:
                raise ValueError(f'score called with invalid y: {y}')
            
            sim_val = 1 if c in psi else -1
            g = get_def_elem()
            
            score_case_1 = self.score_simple(x - 1, y - 1) + sim_val
            score_case_2 = self.score_simple(x - 1, y) + self.GAMMA
            score_case_3 = self.score_simple(x, y - 1) + self.GAMMA

            sc = max(score_case_1, score_case_2, score_case_3)

            g['score'] = sc
            if score_case_1 == sc:
                g['diag'] = True
            if score_case_2 == sc:
                g['left'] = True
            if score_case_3 == sc:
                g['down'] = True

            self.set_G(x, y, g)

        return sc

    def score(self, x, y) -> int:
        if x == 0 or y == 0: 
            return -100000000
        
        sc = self.get_G(x, y)['score'] if x in self.G and y in self.G[x] else None
        if sc is None:
            c = self.get_G(x, 0)['c']
            if c is None:
                raise ValueError(f'score called with invalid x: {x}')
            psi = self.get_G(0, y)['psi']
            if psi is None:
                raise ValueError(f'score called with invalid y: {y}')

            sim_val = self.sim(c, psi)
            g = get_def_elem()

            score_case_1 = self.score(x - 1, y - 1) + sim_val
            score_case_2 = self.score(x - 1, y) + (min(0, sim_val) if x != 1 and y != 1 else self.GAMMA)
            score_case_3 = self.score(x, y - 1) + self.GAMMA

            sc = max(score_case_1, score_case_2, score_case_3)

            g['score'] = sc
            if score_case_1 == sc:
                g['diag'] = True
            if score_case_2 == sc:
                g['left'] = True
            if score_case_3 == sc:
                g['down'] = True

            self.set_G(x, y, g)
        return sc



if __name__ == "__main__":
    # P = ['A', 'C', 'B', 'A', 'C', 'A', 'D']
    # Psi = [{'A'}, {'A', 'B', 'C'}, {'A', 'C'}, 'D']

    P = ['A', 'B', 'C', 'D', 'A', 'B', 'E']
    Psi = [{'A'}, {'C'}, {'D'}, {'D'}, {'C'}, {'B'}, {'C'}]
    asm = ASM(P, Psi, True)
    print(asm)