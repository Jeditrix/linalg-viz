"""
linalg-viz: A linear algebra visualization library for learning.

Visualize vectors, matrices, and linear transformations with intuitive animations.

Quick Start:
    from linalg_viz import Vector, Matrix, show

    # Display vectors
    show(Vector(1, 2), Vector(3, 1))

    # Animate a linear transformation
    M = Matrix([[2, 0], [0, 2]])
    Vector(1, 1).transform(M).animate().show()

    # Visualize matrix multiplication step-by-step
    from linalg_viz import MatrixScene
    import numpy as np

    A = np.array([[1, 2], [3, 4]])
    v = np.array([1, 2])
    MatrixScene().show_matrix_vector_multiply(A, v)
"""

from linalg_viz.core.vector import Vector
from linalg_viz.core.matrix import Matrix
from linalg_viz.scene.scene import Scene
from linalg_viz.scene.matrix_scene import MatrixScene
from linalg_viz.rendering.colors import Colors


def show(*vectors: Vector) -> None:
    """Display vectors in a window.

    Args:
        *vectors: One or more vectors to display (can be animated)

    Example:
        show(Vector(1, 2), Vector(3, 4))
        show(v1.transform(M).animate(), v2.transform(M).animate())
    """
    if not vectors:
        return

    dim = vectors[0].dim
    scene = Scene(dim=dim)

    has_animation = False
    for v in vectors:
        scene.add(v)
        if v._pending_animation is not None:
            scene._add_animation(v._pending_animation)
            has_animation = True

    if has_animation:
        scene.play()
    else:
        scene.show()


__version__ = "0.1.0"
__all__ = ["Vector", "Matrix", "Scene", "MatrixScene", "Colors", "show"]
