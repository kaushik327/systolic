from dataclasses import dataclass
import numpy as np
from itertools import product

# algorithm from https://ecelabs.njit.edu/ece459/lab3.php


@dataclass
class SystolicUnit:
    in_a: int = 0
    in_b: int = 0
    out_a: int = 0
    out_b: int = 0
    c: int = 0


class SystolicArray:
    def __init__(self):
        self.array = [[SystolicUnit() for _ in range(4)] for _ in range(4)]

    def update_inputs(self, aa: list[int], bb: list[int]):
        """
        Pipe numbers from one cell's output to the next cell's input,
        or from the input arrays to the topmost/leftmost cells' inputs
        """
        for r, c in product(range(4), range(4)):
            if c == 0:
                self.array[r][c].in_a = aa[r]
            else:
                self.array[r][c].in_a = self.array[r][c - 1].out_a
            if r == 0:
                self.array[r][c].in_b = bb[c]
            else:
                self.array[r][c].in_b = self.array[r - 1][c].out_b

    def run_cells(self):
        for r, c in product(range(4), range(4)):
            self.array[r][c].c += self.array[r][c].in_a * self.array[r][c].in_b
            self.array[r][c].out_a = self.array[r][c].in_a
            self.array[r][c].out_b = self.array[r][c].in_b

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
