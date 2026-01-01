"""Animation classes."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Tuple
import numpy as np

from linalg_viz.animation.easing import Easing, EasingFunc

if TYPE_CHECKING:
    from linalg_viz.core.vector import Vector
    from linalg_viz.core.matrix import Matrix


class Animation:
    """Base animation class."""

    def __init__(self, duration: float = 1.0, easing: EasingFunc = None):
        self._duration = duration
        self._easing = easing or Easing.ease_in_out_cubic
        self._elapsed = 0.0
        self._finished = False

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def progress(self) -> float:
        return min(1.0, self._elapsed / self._duration) if self._duration else 1.0

    @property
    def eased_progress(self) -> float:
        return self._easing(self.progress)

    @property
    def is_finished(self) -> bool:
        return self._finished

    def update(self, dt: float) -> None:
        if self._finished:
            return
        self._elapsed += dt
        if self._elapsed >= self._duration:
            self._elapsed = self._duration
            self._finished = True

    def reset(self) -> None:
        self._elapsed = 0.0
        self._finished = False


class VectorAnimation(Animation):
    """Interpolates a vector between two states."""

    def __init__(self, start: 'Vector', end: 'Vector', duration: float = 1.0, easing: EasingFunc = None):
        super().__init__(duration, easing)
        self._start = start
        self._end = end

    def get_value(self) -> Tuple[np.ndarray, np.ndarray]:
        t = self.eased_progress
        components = self._start._components + t * (self._end._components - self._start._components)
        origin = self._start._origin + t * (self._end._origin - self._start._origin)
        return (components, origin)


class GridTransformAnimation(Animation):
    """Shows a grid transforming under a matrix."""

    def __init__(self, matrix: 'Matrix', duration: float = 2.0, easing: EasingFunc = None):
        super().__init__(duration, easing)
        self._matrix = matrix
        self._identity = np.eye(matrix.dim)

    def get_value(self) -> np.ndarray:
        t = self.eased_progress
        return self._identity + t * (self._matrix._data - self._identity)

    def get_grid_points(self, bounds=(-5, -5, 5, 5), density=10) -> List[Tuple[np.ndarray, np.ndarray]]:
        min_x, min_y, max_x, max_y = bounds
        matrix = self.get_value()
        lines = []
        for i in range(density + 1):
            x = min_x + (i / density) * (max_x - min_x)
            lines.append((matrix @ np.array([x, min_y]), matrix @ np.array([x, max_y])))
        for i in range(density + 1):
            y = min_y + (i / density) * (max_y - min_y)
            lines.append((matrix @ np.array([min_x, y]), matrix @ np.array([max_x, y])))
        return lines

    def get_grid_points_3d(self, bounds=(-5, -5, -5, 5, 5, 5), density=10) -> List[Tuple[np.ndarray, np.ndarray]]:
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        matrix = self.get_value()
        lines = []
        # XZ plane grid (y=0)
        for i in range(density + 1):
            x = min_x + (i / density) * (max_x - min_x)
            lines.append((matrix @ np.array([x, 0, min_z]), matrix @ np.array([x, 0, max_z])))
        for i in range(density + 1):
            z = min_z + (i / density) * (max_z - min_z)
            lines.append((matrix @ np.array([min_x, 0, z]), matrix @ np.array([max_x, 0, z])))
        return lines
