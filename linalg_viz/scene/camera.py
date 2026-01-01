"""Camera classes for 2D and 3D scene navigation."""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np


class Camera2D:
    """2D camera with pan and zoom controls."""

    def __init__(self, width: int = 800, height: int = 600):
        """Initialize the 2D camera.

        Args:
            width: Viewport width in pixels
            height: Viewport height in pixels
        """
        self._width = width
        self._height = height

        self._position = np.array([0.0, 0.0])
        self._zoom = 50.0  # Pixels per unit
        self._rotation = 0.0  # Radians

    @property
    def position(self) -> Tuple[float, float]:
        """Camera position (center of view in world coordinates)."""
        return tuple(self._position)

    @property
    def zoom(self) -> float:
        """Zoom level (pixels per unit)."""
        return self._zoom

    @property
    def rotation(self) -> float:
        """Camera rotation in radians."""
        return self._rotation

    def resize(self, width: int, height: int) -> None:
        """Update viewport dimensions."""
        self._width = width
        self._height = height

    def pan(self, dx: float, dy: float) -> None:
        """Pan the camera by screen pixels.

        Args:
            dx: Horizontal pan in pixels
            dy: Vertical pan in pixels
        """
        self._position[0] -= dx / self._zoom
        self._position[1] += dy / self._zoom

    def zoom_by(self, factor: float, center: Tuple[float, float] = None) -> None:
        """Zoom by a factor.

        Args:
            factor: Zoom multiplier (>1 to zoom in, <1 to zoom out)
            center: Screen position to zoom toward (defaults to center)
        """
        old_zoom = self._zoom
        self._zoom = max(5.0, min(500.0, self._zoom * factor))

        if center is not None:
            cx, cy = center
            world_x = (cx - self._width / 2) / old_zoom + self._position[0]
            world_y = -(cy - self._height / 2) / old_zoom + self._position[1]

            new_screen_x = (world_x - self._position[0]) * self._zoom + self._width / 2
            new_screen_y = -(world_y - self._position[1]) * self._zoom + self._height / 2

            self._position[0] += (new_screen_x - cx) / self._zoom
            self._position[1] -= (new_screen_y - cy) / self._zoom

    def rotate(self, angle: float) -> None:
        """Rotate the camera view.

        Args:
            angle: Rotation angle in radians
        """
        self._rotation += angle

    def reset(self) -> None:
        """Reset camera to default position."""
        self._position = np.array([0.0, 0.0])
        self._zoom = 50.0
        self._rotation = 0.0

    def world_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates to screen coordinates.

        Args:
            x: World X coordinate
            y: World Y coordinate

        Returns:
            (screen_x, screen_y) tuple
        """
        if self._rotation != 0:
            c, s = math.cos(-self._rotation), math.sin(-self._rotation)
            rx = x * c - y * s
            ry = x * s + y * c
            x, y = rx, ry

        screen_x = (x - self._position[0]) * self._zoom + self._width / 2
        screen_y = -(y - self._position[1]) * self._zoom + self._height / 2
        return (screen_x, screen_y)

    def screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates.

        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate

        Returns:
            (world_x, world_y) tuple
        """
        x = (screen_x - self._width / 2) / self._zoom + self._position[0]
        y = -(screen_y - self._height / 2) / self._zoom + self._position[1]

        if self._rotation != 0:
            c, s = math.cos(self._rotation), math.sin(self._rotation)
            rx = x * c - y * s
            ry = x * s + y * c
            x, y = rx, ry

        return (x, y)

    def get_view_bounds(self) -> Tuple[float, float, float, float]:
        """Get the visible world bounds.

        Returns:
            (min_x, min_y, max_x, max_y) tuple
        """
        half_w = self._width / 2 / self._zoom
        half_h = self._height / 2 / self._zoom
        return (
            self._position[0] - half_w,
            self._position[1] - half_h,
            self._position[0] + half_w,
            self._position[1] + half_h,
        )


class Camera3D:
    """3D camera with orbit controls."""

    def __init__(self, width: int = 800, height: int = 600):
        """Initialize the 3D camera.

        Args:
            width: Viewport width in pixels
            height: Viewport height in pixels
        """
        self._width = width
        self._height = height

        self._target = np.array([0.0, 0.0, 0.0])
        self._distance = 10.0
        self._theta = math.pi / 4  # Azimuth angle
        self._phi = math.pi / 6  # Elevation angle

        self._fov = 45.0  # Field of view in degrees
        self._near = 0.1
        self._far = 1000.0

    @property
    def position(self) -> Tuple[float, float, float]:
        """Camera position in world coordinates."""
        x = self._target[0] + self._distance * math.cos(self._phi) * math.cos(self._theta)
        y = self._target[1] + self._distance * math.sin(self._phi)
        z = self._target[2] + self._distance * math.cos(self._phi) * math.sin(self._theta)
        return (x, y, z)

    @property
    def target(self) -> Tuple[float, float, float]:
        """Camera look-at target."""
        return tuple(self._target)

    def resize(self, width: int, height: int) -> None:
        """Update viewport dimensions."""
        self._width = width
        self._height = height

    def orbit(self, d_theta: float, d_phi: float) -> None:
        """Orbit the camera around the target.

        Args:
            d_theta: Change in azimuth angle (horizontal rotation)
            d_phi: Change in elevation angle (vertical rotation)
        """
        self._theta += d_theta
        self._phi = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self._phi + d_phi))

    def pan(self, dx: float, dy: float) -> None:
        """Pan the camera target.

        Args:
            dx: Horizontal pan
            dy: Vertical pan
        """
        right = np.array([math.cos(self._theta + math.pi / 2), 0, math.sin(self._theta + math.pi / 2)])
        up = np.array([0, 1, 0])

        scale = self._distance * 0.005
        self._target += -right * dx * scale + up * dy * scale

    def zoom_by(self, factor: float) -> None:
        """Zoom by adjusting distance to target.

        Args:
            factor: Zoom multiplier (>1 to zoom out, <1 to zoom in)
        """
        self._distance = max(1.0, min(100.0, self._distance * factor))

    def reset(self) -> None:
        """Reset camera to default position."""
        self._target = np.array([0.0, 0.0, 0.0])
        self._distance = 10.0
        self._theta = math.pi / 4
        self._phi = math.pi / 6

    def get_view_matrix(self) -> np.ndarray:
        """Get the view matrix for OpenGL."""
        pos = np.array(self.position)
        target = self._target

        forward = target - pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -np.array([np.dot(right, pos), np.dot(up, pos), np.dot(-forward, pos)])

        return view

    def get_projection_matrix(self) -> np.ndarray:
        """Get the perspective projection matrix."""
        aspect = self._width / self._height
        fov_rad = math.radians(self._fov)
        f = 1.0 / math.tan(fov_rad / 2)

        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (self._far + self._near) / (self._near - self._far)
        proj[2, 3] = (2 * self._far * self._near) / (self._near - self._far)
        proj[3, 2] = -1

        return proj
