"""Generate GIFs for README documentation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from linalg_viz import MatrixScene

# Create assets folder
assets = Path(__file__).parent / "assets"
assets.mkdir(exist_ok=True)

print("Generating GIFs...")

scene = MatrixScene()

# Matrix-vector multiplication
M = np.array([
    [2, -1, 3],
    [0, 4, 1],
    [1, 2, -2]
])
v = np.array([1, 2, 3])
scene.record_matrix_vector_multiply(M, v, str(assets / "matrix_vector.gif"))

# Matrix-matrix multiplication
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])
scene = MatrixScene(width=1200)
scene.record_matrix_multiply(A, B, str(assets / "matrix_multiply.gif"))

# Dot product
a = np.array([2, 3, 4])
b = np.array([1, -2, 3])
scene = MatrixScene()
scene.record_dot_product(a, b, str(assets / "dot_product.gif"))

print("Done! GIFs saved to assets/")
