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


def shift_inputs_for_matmul(A: np.ndarray, B: np.ndarray):
    N = A.shape[0]
    assert A.shape == B.shape == (N, N)

    padded_A = np.pad(A, ((0, 0), (2 * N - 2, N - 1)))
    padded_B = np.pad(B, ((2 * N - 2, N - 1), (0, 0)))

    for i in reversed(range(3 * N - 2)):
        yield (
            np.diagonal(padded_A, offset=i),
            np.diagonal(padded_B, offset=-i),
        )


class SystolicArray:
    def __init__(self, N: int):
        assert N > 1
        self.array = [[SystolicUnit() for _ in range(N)] for _ in range(N)]
        self.N = N

        # "Wiring" the units together in a grid
        for r, c in product(range(N), range(N)):
            if c + 1 < N:
                self.array[r][c].out_a = self.array[r][c + 1]
            if r + 1 < N:
                self.array[r][c].out_b = self.array[r + 1][c]

    def update_inputs(self, aa: list[int], bb: list[int]):
        """Update inputs of leftmost/uppermost cells"""
        for r in range(self.N):
            self.array[r][0].a = aa[r]
        for c in range(self.N):
            self.array[0][c].b = bb[c]

    def run_cells(self):
        # Running in this exact order (large to small r and c) is important
        # so values aren't moved multiple times in one cycle
        for r, c in product(reversed(range(self.N)), reversed(range(self.N))):
            cell = self.array[r][c]
            cell.c += cell.a * cell.b
            if cell.out_a:
                cell.out_a.a = cell.a
            if cell.out_b:
                cell.out_b.b = cell.b

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        assert A.shape == B.shape == (self.N, self.N)
        for a, b in shift_inputs_for_matmul(A, B):
            self.update_inputs(a, b)
            self.run_cells()

        return self.vals()

    def vals(self):
        ret = np.zeros((self.N, self.N), dtype=int)
        for r, c in product(range(self.N), range(self.N)):
            ret[r, c] = self.array[r][c].c
        return ret


# Example usage
if __name__ == "__main__":
    N = 10

    A = np.random.randint(-10, 10, (N, N))
    B = np.random.randint(-10, 10, (N, N))

    systolic_array = SystolicArray(N=N)
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
