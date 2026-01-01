"""Matrix class for linear algebra visualization."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    from linalg_viz.core.vector import Vector


class Matrix:
    """2x2 or 3x3 matrix for linear transformations."""

    def __init__(self, data: Union[List[List[float]], np.ndarray]):
        self._data = np.array(data, dtype=np.float64)
        if self._data.ndim != 2 or self._data.shape[0] != self._data.shape[1]:
            raise ValueError("Matrix must be square")
        if self._data.shape[0] not in (2, 3):
            raise ValueError("Matrix must be 2x2 or 3x3")
        self._dim = self._data.shape[0]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def data(self) -> np.ndarray:
        return self._data.copy()

    @classmethod
    def identity(cls, dim: int = 2) -> Matrix:
        return cls(np.eye(dim))

    @classmethod
    def rotation(cls, angle: float) -> Matrix:
        c, s = np.cos(angle), np.sin(angle)
        return cls([[c, -s], [s, c]])

    @classmethod
    def rotation_x(cls, angle: float) -> Matrix:
        c, s = np.cos(angle), np.sin(angle)
        return cls([[1, 0, 0], [0, c, -s], [0, s, c]])

    @classmethod
    def rotation_y(cls, angle: float) -> Matrix:
        c, s = np.cos(angle), np.sin(angle)
        return cls([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @classmethod
    def rotation_z(cls, angle: float) -> Matrix:
        c, s = np.cos(angle), np.sin(angle)
        return cls([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @classmethod
    def scaling(cls, *factors: float) -> Matrix:
        return cls(np.diag(factors))

    @classmethod
    def shear(cls, shear_x: float = 0, shear_y: float = 0) -> Matrix:
        return cls([[1, shear_x], [shear_y, 1]])

    @classmethod
    def reflection(cls, axis: str = "x") -> Matrix:
        if axis == "x":
            return cls([[1, 0], [0, -1]])
        elif axis == "y":
            return cls([[-1, 0], [0, 1]])
        return cls([[-1, 0], [0, -1]])

    @classmethod
    def projection(cls, onto: Tuple[float, float]) -> Matrix:
        x, y = onto
        n = x * x + y * y
        return cls([[x * x / n, x * y / n], [x * y / n, y * y / n]])

    def determinant(self) -> float:
        return float(np.linalg.det(self._data))

    def trace(self) -> float:
        return float(np.trace(self._data))

    def inverse(self) -> Matrix:
        return Matrix(np.linalg.inv(self._data))

    def transpose(self) -> Matrix:
        return Matrix(self._data.T)

    def eigenvalues(self) -> List[complex]:
        return [complex(v) for v in np.linalg.eigvals(self._data)]

    def eigenvectors(self) -> List[Tuple[complex, 'Vector']]:
        from linalg_viz.core.vector import Vector
        vals, vecs = np.linalg.eig(self._data)
        return [(complex(vals[i]), Vector(*vecs[:, i].real)) for i in range(len(vals))]

    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self._data))

    def __repr__(self) -> str:
        rows = [", ".join(f"{x:.3f}" for x in row) for row in self._data]
        return f"Matrix([{'], ['.join(rows)}])"

    def __matmul__(self, other: Matrix) -> Matrix:
        return Matrix(self._data @ other._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False
        return np.allclose(self._data, other._data)
