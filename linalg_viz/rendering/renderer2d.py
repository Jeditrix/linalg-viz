"""2D OpenGL renderer."""

from __future__ import annotations
from typing import TYPE_CHECKING
import math

import pygame
from OpenGL.GL import *

from linalg_viz.rendering.colors import Colors, RGBA

if TYPE_CHECKING:
    from linalg_viz.core.vector import Vector
    from linalg_viz.scene.camera import Camera2D
    from linalg_viz.scene.grid import Grid2D
    from linalg_viz.animation.animator import VectorAnimation, GridTransformAnimation


class Renderer2D:
    """OpenGL 2D renderer."""

    def __init__(self, width: int = 800, height: int = 600):
        self._width = width
        self._height = height
        self._font = None
        self._font_small = None

    def init_gl(self) -> None:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # DISABLE line smoothing - it causes color bleeding/halos when combined with
        # alpha blending because it modulates edge pixel alpha values which then
        # blend incorrectly with nearby primitives of different colors.
        glDisable(GL_LINE_SMOOTH)
        # Enable multisampling (MSAA) for proper anti-aliasing without color bleeding
        # The multisample buffers are set up in pygame display initialization
        glEnable(GL_MULTISAMPLE)
        glLineWidth(2.0)  # Slightly thicker lines look better with MSAA
        pygame.font.init()
        self._font = pygame.font.SysFont('monospace', 16)
        self._font_small = pygame.font.SysFont('monospace', 12)

    def resize(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        glViewport(0, 0, width, height)

    def clear(self, color: RGBA = None) -> None:
        if color is None:
            color = Colors.BACKGROUND
        glClearColor(*color)
        glClear(GL_COLOR_BUFFER_BIT)

    def setup_2d_projection(self, camera: 'Camera2D') -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self._width, self._height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def draw_line(self, x1: float, y1: float, x2: float, y2: float,
                  camera: 'Camera2D', color: RGBA = None, width: float = 1.0) -> None:
        if color is None:
            color = Colors.WHITE
        # Ensure clean OpenGL state for line rendering
        glDisable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        sx1, sy1 = camera.world_to_screen(x1, y1)
        sx2, sy2 = camera.world_to_screen(x2, y2)
        glLineWidth(width)
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex2f(sx1, sy1)
        glVertex2f(sx2, sy2)
        glEnd()

    def draw_grid(self, grid: 'Grid2D', camera: 'Camera2D') -> None:
        for (x1, y1), (x2, y2), is_major in grid.get_grid_lines(camera):
            color = Colors.GRID_MAJOR if is_major else Colors.GRID
            width = 1.5 if is_major else 1.0
            self.draw_line(x1, y1, x2, y2, camera, color, width)
        for (x1, y1), (x2, y2), axis in grid.get_axis_lines(camera):
            color = Colors.AXIS_X if axis == 'x' else Colors.AXIS_Y
            self.draw_line(x1, y1, x2, y2, camera, color, 2.0)

    def draw_arrow(self, origin_x: float, origin_y: float, end_x: float, end_y: float,
                   camera: 'Camera2D', color: RGBA = None, width: float = 2.5) -> None:
        if color is None:
            color = Colors.RED
        self.draw_line(origin_x, origin_y, end_x, end_y, camera, color, width)

        dx, dy = end_x - origin_x, end_y - origin_y
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return

        head_length = min(0.15 * length, 0.3)
        angle = math.atan2(dy, dx)
        left_x = end_x + head_length * math.cos(angle + math.pi - 0.4)
        left_y = end_y + head_length * math.sin(angle + math.pi - 0.4)
        right_x = end_x + head_length * math.cos(angle + math.pi + 0.4)
        right_y = end_y + head_length * math.sin(angle + math.pi + 0.4)

        sx_end, sy_end = camera.world_to_screen(end_x, end_y)
        sx_left, sy_left = camera.world_to_screen(left_x, left_y)
        sx_right, sy_right = camera.world_to_screen(right_x, right_y)

        # Ensure clean OpenGL state for triangle rendering
        glDisable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        glColor4f(*color)
        glBegin(GL_TRIANGLES)
        glVertex2f(sx_end, sy_end)
        glVertex2f(sx_left, sy_left)
        glVertex2f(sx_right, sy_right)
        glEnd()

    def draw_vector(self, vector: 'Vector', camera: 'Camera2D', color: RGBA = None) -> None:
        if color is None:
            color = vector._color
        ox, oy = vector.origin[0], vector.origin[1]
        self.draw_arrow(ox, oy, ox + vector.x, oy + vector.y, camera, color)

    def draw_animated_vector(self, animation: 'VectorAnimation', camera: 'Camera2D') -> None:
        components, origin = animation.get_value()

        # Ghost of original - use dimmed opaque color for consistency with 3D
        original_color = animation._start._color
        ghost_color = (
            original_color[0] * 0.4,
            original_color[1] * 0.4,
            original_color[2] * 0.4,
            1.0
        )
        ox, oy = animation._start.origin[0], animation._start.origin[1]
        self.draw_arrow(ox, oy, ox + animation._start.x, oy + animation._start.y, camera, ghost_color, 1.5)

        # Current position
        ox, oy = origin[0], origin[1]
        self.draw_arrow(ox, oy, ox + components[0], oy + components[1], camera, animation._end._color)

    def draw_transformed_grid(self, animation: 'GridTransformAnimation', camera: 'Camera2D') -> None:
        for start, end in animation.get_grid_points():
            self.draw_line(start[0], start[1], end[0], end[1], camera, Colors.TRANSFORM_AFTER, 1.0)

    def draw_controls_hint(self, paused: bool) -> None:
        """Draw controls hint overlay."""
        if not self._font:
            return

        # Status text
        status = "PAUSED" if paused else "PLAYING"
        controls = "Space:Pause  R:Replay  ←→:Step  C:Reset  Esc:Exit"

        # Render to pygame surface
        status_surface = self._font.render(status, True, (200, 200, 200) if not paused else (255, 200, 100))
        controls_surface = self._font_small.render(controls, True, (150, 150, 150))

        # Convert to OpenGL texture and draw
        self._draw_text_surface(status_surface, 10, 10)
        self._draw_text_surface(controls_surface, 10, self._height - 25)

    def _draw_text_surface(self, surface: pygame.Surface, x: int, y: int) -> None:
        """Draw a pygame surface as OpenGL texture."""
        text_data = pygame.image.tostring(surface, "RGBA", True)
        w, h = surface.get_size()

        # Create texture
        glEnable(GL_TEXTURE_2D)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Draw quad
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 0); glVertex2f(x, y + h)
        glEnd()

        # Clean up texture state completely to prevent color bleeding
        glBindTexture(GL_TEXTURE_2D, 0)
        glDeleteTextures([texture])
        glDisable(GL_TEXTURE_2D)
        glTexCoord2f(0, 0)  # Reset texture coordinates
        glColor4f(1, 1, 1, 1)  # Reset color to white
