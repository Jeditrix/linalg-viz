"""Linear transformation with animation."""

from linalg_viz import Vector, Matrix, show

M = Matrix([[2, 22], [0, 1.5]])

e1 = Vector(3, 6).color("red")
e2 = Vector(0, 1).color("blue")

# Transform and animate
show(e1.transform(M).animate())
