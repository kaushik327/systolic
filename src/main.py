from dataclasses import dataclass
import numpy as np
from itertools import product

from typing import Optional

# algorithm from https://ecelabs.njit.edu/ece459/lab3.php


@dataclass
class SystolicUnit:
    a: int = 0
    b: int = 0
    out_a: Optional["SystolicUnit"] = None
    out_b: Optional["SystolicUnit"] = None
    c: int = 0


class SystolicArray:
    def __init__(self):
        self.array = [[SystolicUnit() for _ in range(4)] for _ in range(4)]

        # "Wiring" the units together in a grid
        for r, c in product(range(4), range(4)):
            if c + 1 < 4:
                self.array[r][c].out_a = self.array[r][c + 1]
            if r + 1 < 4:
                self.array[r][c].out_b = self.array[r + 1][c]

    def update_inputs(self, aa: list[int], bb: list[int]):
        """Update inputs of leftmost/uppermost cells"""
        for r in range(4):
            self.array[r][0].a = aa[r]
        for c in range(4):
            self.array[0][c].b = bb[c]

    def run_cells(self):
        # Running in this exact order (large to small r and c) is important
        # so values aren't moved multiple times in one cycle
        for r, c in product(reversed(range(4)), reversed(range(4))):
            cell = self.array[r][c]
            cell.c += cell.a * cell.b
            if cell.out_a:
                cell.out_a.a = cell.a
            if cell.out_b:
                cell.out_b.b = cell.b

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        assert A.shape == B.shape == (4, 4)

        for i in range(3, -4, -1):
            if i >= 0:
                self.update_inputs(
                    np.pad(np.diagonal(A, offset=i), (0, i)),
                    np.pad(np.diagonal(B, offset=-i), (0, i)),
                )
            else:
                self.update_inputs(
                    np.pad(np.diagonal(A, offset=i), (-i, 0)),
                    np.pad(np.diagonal(B, offset=-i), (-i, 0)),
                )
            self.run_cells()

        for i in range(3):
            self.update_inputs(np.zeros(4), np.zeros(4))
            self.run_cells()

        return self.vals()

    def vals(self):
        ret = np.zeros((4, 4), dtype=int)
        for r, c in product(range(4), range(4)):
            ret[r, c] = self.array[r][c].c
        return ret


# Example usage
if __name__ == "__main__":
    A = np.random.randint(0, 10, (4, 4))
    B = np.random.randint(0, 10, (4, 4))

    systolic_array = SystolicArray()
    result = systolic_array.matmul(A, B)

    print("Matrix A:")
    print(A)

    print("\nMatrix B:")
    print(B)

    print("\nResult:")
    print(result)

    print("\nExpected:")
    print(np.matmul(A, B))

    assert np.all(result == A @ B)
