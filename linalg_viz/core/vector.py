"""Vector class for linear algebra visualization."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    from linalg_viz.core.matrix import Matrix
    from linalg_viz.scene.scene import Scene
    from linalg_viz.animation.animator import Animation


class Vector:
    """2D or 3D vector with fluent API."""

    def __init__(self, *components: float, origin: Optional[Tuple[float, ...]] = None):
        if len(components) == 1 and hasattr(components[0], '__iter__'):
            components = tuple(components[0])
        if len(components) not in (2, 3):
            raise ValueError("Vector must have 2 or 3 components")

        self._components = np.array(components, dtype=np.float64)
        self._dim = len(components)
        self._origin = np.array(origin, dtype=np.float64) if origin else np.zeros(self._dim)
        self._color = (1.0, 0.3, 0.3, 1.0)
        self._pending_animation: Optional[Animation] = None
        self._previous_state: Optional[Vector] = None
        self._scene: Optional[Scene] = None

    @property
    def x(self) -> float:
        return float(self._components[0])

    @property
    def y(self) -> float:
        return float(self._components[1])

    @property
    def z(self) -> float:
        return float(self._components[2]) if self._dim == 3 else 0.0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def components(self) -> np.ndarray:
        return self._components.copy()

    @property
    def origin(self) -> np.ndarray:
        return self._origin.copy()

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self._components))

    @property
    def normalized(self) -> Vector:
        mag = self.magnitude
        if mag == 0:
            return Vector(*self._components)
        return Vector(*(self._components / mag))

    def copy(self) -> Vector:
        v = Vector(*self._components, origin=tuple(self._origin))
        v._color = self._color
        return v

    def transform(self, matrix: Matrix) -> Vector:
        from linalg_viz.core.matrix import Matrix
        if not isinstance(matrix, Matrix):
            matrix = Matrix(matrix)

        new_components = matrix._data @ self._components
        new_origin = matrix._data @ self._origin if np.any(self._origin) else self._origin

        result = Vector(*new_components, origin=tuple(new_origin))
        result._color = self._color
        result._previous_state = self.copy()
        result._scene = self._scene
        return result

    def scale(self, factor: float) -> Vector:
        result = Vector(*(self._components * factor), origin=tuple(self._origin))
        result._color = self._color
        result._previous_state = self.copy()
        result._scene = self._scene
        return result

    def add(self, other: Vector) -> Vector:
        if self._dim != other._dim:
            raise ValueError("Cannot add vectors of different dimensions")
        result = Vector(*(self._components + other._components), origin=tuple(self._origin))
        result._color = self._color
        result._previous_state = self.copy()
        result._scene = self._scene
        return result

    def subtract(self, other: Vector) -> Vector:
        if self._dim != other._dim:
            raise ValueError("Cannot subtract vectors of different dimensions")
        result = Vector(*(self._components - other._components), origin=tuple(self._origin))
        result._color = self._color
        result._previous_state = self.copy()
        result._scene = self._scene
        return result

    def dot(self, other: Vector) -> float:
        if self._dim != other._dim:
            raise ValueError("Cannot dot vectors of different dimensions")
        return float(np.dot(self._components, other._components))

    def cross(self, other: Vector) -> Vector:
        if self._dim != 3 or other._dim != 3:
            raise ValueError("Cross product requires 3D vectors")
        return Vector(*np.cross(self._components, other._components))

    def project_onto(self, other: Vector) -> Vector:
        """Project this vector onto another vector."""
        if self._dim != other._dim:
            raise ValueError("Cannot project vectors of different dimensions")
        other_mag_sq = np.dot(other._components, other._components)
        if other_mag_sq < 1e-10:
            return Vector(*np.zeros(self._dim))
        scalar = np.dot(self._components, other._components) / other_mag_sq
        result = Vector(*(scalar * other._components), origin=tuple(self._origin))
        result._color = self._color
        result._previous_state = self.copy()
        result._scene = self._scene
        return result

    def angle_with(self, other: Vector) -> float:
        """Return angle between vectors in radians."""
        if self._dim != other._dim:
            raise ValueError("Cannot compute angle between vectors of different dimensions")
        cos_angle = np.dot(self._components, other._components) / (self.magnitude * other.magnitude + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def color(self, c: Union[str, Tuple[float, ...]]) -> Vector:
        from linalg_viz.rendering.colors import Colors
        if isinstance(c, str):
            self._color = Colors.get(c)
        elif len(c) == 3:
            self._color = (c[0], c[1], c[2], 1.0)
        else:
            self._color = tuple(c)
        return self

    def at(self, origin: Tuple[float, ...]) -> Vector:
        self._origin = np.array(origin, dtype=np.float64)
        return self

    def animate(self, duration: float = 1.0) -> Vector:
        from linalg_viz.animation.animator import VectorAnimation
        if self._previous_state is not None:
            self._pending_animation = VectorAnimation(self._previous_state, self, duration)
        return self

    def show(self, scene: Optional[Scene] = None) -> None:
        from linalg_viz.scene.scene import Scene
        if scene is None:
            scene = Scene(dim=self._dim)
        scene.add(self)
        if self._pending_animation is not None:
            scene._add_animation(self._pending_animation)
            scene.play()
        else:
            scene.show()

    def __repr__(self) -> str:
        if self._dim == 2:
            return f"Vector({self.x:.3f}, {self.y:.3f})"
        return f"Vector({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return False
        return np.allclose(self._components, other._components)

    def __add__(self, other: Vector) -> Vector:
        return self.add(other)

    def __sub__(self, other: Vector) -> Vector:
        return self.subtract(other)

    def __mul__(self, scalar: float) -> Vector:
        return self.scale(scalar)

    def __rmul__(self, scalar: float) -> Vector:
        return self.scale(scalar)

    def __neg__(self) -> Vector:
        return self.scale(-1)
