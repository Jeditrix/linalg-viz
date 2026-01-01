"""Grid and axis rendering for the Cartesian plane."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple
import math

if TYPE_CHECKING:
    from linalg_viz.scene.camera import Camera2D, Camera3D


class Grid2D:
    """2D Cartesian grid with axes."""

    def __init__(self):
        self._show_grid = True
        self._show_axes = True
        self._show_labels = True
        self._grid_spacing = 1.0
        self._major_every = 5

    def get_grid_lines(self, camera: 'Camera2D') -> List[Tuple[Tuple[float, float], Tuple[float, float], bool]]:
        """Get grid lines to draw.

        Args:
            camera: The 2D camera for determining visible bounds

        Returns:
            List of ((x1, y1), (x2, y2), is_major) tuples
        """
        if not self._show_grid:
            return []

        min_x, min_y, max_x, max_y = camera.get_view_bounds()

        spacing = self._grid_spacing
        zoom = camera.zoom

        if zoom < 20:
            spacing = 5.0
        elif zoom < 50:
            spacing = 2.0
        elif zoom > 150:
            spacing = 0.5
        elif zoom > 300:
            spacing = 0.2

        lines = []

        start_x = math.floor(min_x / spacing) * spacing
        x = start_x
        while x <= max_x:
            is_major = abs(x) < 0.001 or abs(round(x / spacing) % self._major_every) < 0.001
            lines.append(((x, min_y), (x, max_y), is_major))
            x += spacing

        start_y = math.floor(min_y / spacing) * spacing
        y = start_y
        while y <= max_y:
            is_major = abs(y) < 0.001 or abs(round(y / spacing) % self._major_every) < 0.001
            lines.append(((min_x, y), (max_x, y), is_major))
            y += spacing

        return lines

    def get_axis_lines(self, camera: 'Camera2D') -> List[Tuple[Tuple[float, float], Tuple[float, float], str]]:
        """Get axis lines to draw.

        Args:
            camera: The 2D camera for determining visible bounds

        Returns:
            List of ((x1, y1), (x2, y2), axis) tuples where axis is 'x' or 'y'
        """
        if not self._show_axes:
            return []

        min_x, min_y, max_x, max_y = camera.get_view_bounds()

        return [
            ((min_x, 0), (max_x, 0), 'x'),
            ((0, min_y), (0, max_y), 'y'),
        ]

    def get_axis_labels(self, camera: 'Camera2D') -> List[Tuple[float, float, str]]:
        """Get axis tick labels.

        Args:
            camera: The 2D camera for determining visible bounds

        Returns:
            List of (x, y, text) tuples
        """
        if not self._show_labels:
            return []

        min_x, min_y, max_x, max_y = camera.get_view_bounds()
        zoom = camera.zoom

        spacing = 1.0
        if zoom < 20:
            spacing = 5.0
        elif zoom < 50:
            spacing = 2.0
        elif zoom > 150:
            spacing = 0.5

        labels = []

        start_x = math.floor(min_x / spacing) * spacing
        x = start_x
        while x <= max_x:
            if abs(x) > 0.001:
                label = f"{x:.0f}" if x == int(x) else f"{x:.1f}"
                labels.append((x, -0.3, label))
            x += spacing

        start_y = math.floor(min_y / spacing) * spacing
        y = start_y
        while y <= max_y:
            if abs(y) > 0.001:
                label = f"{y:.0f}" if y == int(y) else f"{y:.1f}"
                labels.append((-0.3, y, label))
            y += spacing

        return labels


class Grid3D:
    """3D Cartesian grid with axes."""

    def __init__(self):
        self._show_grid = True
        self._show_axes = True
        self._grid_size = 10
        self._grid_spacing = 1.0

    def get_grid_lines(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Get grid lines on the XZ plane."""
        if not self._show_grid:
            return []

        lines = []
        half = self._grid_size / 2

        x = -half
        while x <= half:
            lines.append(((x, 0, -half), (x, 0, half)))
            x += self._grid_spacing

        z = -half
        while z <= half:
            lines.append(((-half, 0, z), (half, 0, z)))
            z += self._grid_spacing

        return lines

    def get_axis_lines(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], str]]:
        """Get axis lines."""
        if not self._show_axes:
            return []

        length = self._grid_size / 2 + 1

        return [
            ((-length, 0, 0), (length, 0, 0), 'x'),
            ((0, -length, 0), (0, length, 0), 'y'),
            ((0, 0, -length), (0, 0, length), 'z'),
        ]
