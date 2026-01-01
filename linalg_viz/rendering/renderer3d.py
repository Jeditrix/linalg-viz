"""3D OpenGL renderer."""

from __future__ import annotations
from typing import TYPE_CHECKING
import math

import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

from linalg_viz.rendering.colors import Colors, RGBA

if TYPE_CHECKING:
    from linalg_viz.core.vector import Vector
    from linalg_viz.scene.camera import Camera3D
    from linalg_viz.scene.grid import Grid3D
    from linalg_viz.animation.animator import VectorAnimation, GridTransformAnimation


class Renderer3D:
    """OpenGL 3D renderer."""

    def __init__(self, width: int = 800, height: int = 600):
        self._width = width
        self._height = height
        self._font = None
        self._font_small = None

    def init_gl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # DISABLE GL_LINE_SMOOTH - it causes color bleeding/halos with alpha blending
        glDisable(GL_LINE_SMOOTH)
        # Use MSAA instead (configured in scene.py pygame display attributes)
        glEnable(GL_MULTISAMPLE)
        glLineWidth(2.0)
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
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def setup_3d_projection(self, camera: 'Camera3D') -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self._width / self._height, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        pos = camera.position
        target = camera.target
        gluLookAt(pos[0], pos[1], pos[2],
                  target[0], target[1], target[2],
                  0, 1, 0)

    def draw_line_3d(self, x1: float, y1: float, z1: float,
                     x2: float, y2: float, z2: float,
                     color: RGBA = None, width: float = 1.0) -> None:
        if color is None:
            color = Colors.WHITE
        glLineWidth(width)
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex3f(x1, y1, z1)
        glVertex3f(x2, y2, z2)
        glEnd()

    def draw_grid(self, grid: 'Grid3D') -> None:
        # Draw all grid lines in uniform gray - no colored axes
        for (x1, y1, z1), (x2, y2, z2) in grid.get_grid_lines():
            self.draw_line_3d(x1, y1, z1, x2, y2, z2, Colors.GRID, 1.0)

        # Draw axis lines in same gray as grid (no color)
        for (x1, y1, z1), (x2, y2, z2), axis in grid.get_axis_lines():
            self.draw_line_3d(x1, y1, z1, x2, y2, z2, Colors.GRID, 1.5)

    def _get_perpendiculars(self, dx: float, dy: float, dz: float):
        """Get two perpendicular vectors to a direction vector."""
        if abs(dy) < 0.9:
            perp1 = (-dz, 0.0, dx)
        else:
            perp1 = (1.0, 0.0, 0.0)

        p1_len = math.sqrt(perp1[0]**2 + perp1[1]**2 + perp1[2]**2)
        if p1_len < 0.001:
            perp1 = (1.0, 0.0, 0.0)
            p1_len = 1.0
        perp1 = (perp1[0]/p1_len, perp1[1]/p1_len, perp1[2]/p1_len)

        perp2 = (
            dy * perp1[2] - dz * perp1[1],
            dz * perp1[0] - dx * perp1[2],
            dx * perp1[1] - dy * perp1[0]
        )
        return perp1, perp2

    def draw_arrow_3d(self, ox: float, oy: float, oz: float,
                      ex: float, ey: float, ez: float,
                      color: RGBA = None, shaft_radius: float = 0.04) -> None:
        """Draw a 3D arrow as solid geometry (cylinder shaft + cone head).

        Colors are confined to the actual polygon surfaces - no bleeding possible.
        """
        if color is None:
            color = Colors.RED

        dx, dy, dz = ex - ox, ey - oy, ez - oz
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 0.001:
            return

        # Normalize direction
        dx, dy, dz = dx/length, dy/length, dz/length

        # Arrow dimensions - proportional to length
        head_length = min(0.25 * length, 0.3)
        head_radius = shaft_radius * 3.0
        segments = 12

        # Get perpendicular vectors for building cylinder/cone
        perp1, perp2 = self._get_perpendiculars(dx, dy, dz)

        # Shaft end point (where cone base starts)
        shaft_ex = ex - dx * head_length
        shaft_ey = ey - dy * head_length
        shaft_ez = ez - dz * head_length

        glDisable(GL_DEPTH_TEST)
        glColor4f(*color)

        # === Draw cylinder shaft as quad strip ===
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            # Normal direction (for lighting, though we don't use it)
            nx = cos_a * perp1[0] + sin_a * perp2[0]
            ny = cos_a * perp1[1] + sin_a * perp2[1]
            nz = cos_a * perp1[2] + sin_a * perp2[2]

            # Point on origin end of shaft
            p1x = ox + shaft_radius * nx
            p1y = oy + shaft_radius * ny
            p1z = oz + shaft_radius * nz

            # Point on tip end of shaft
            p2x = shaft_ex + shaft_radius * nx
            p2y = shaft_ey + shaft_radius * ny
            p2z = shaft_ez + shaft_radius * nz

            glVertex3f(p1x, p1y, p1z)
            glVertex3f(p2x, p2y, p2z)
        glEnd()

        # === Draw cone head as triangle fan ===
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(ex, ey, ez)  # Tip of arrow
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            px = shaft_ex + head_radius * (cos_a * perp1[0] + sin_a * perp2[0])
            py = shaft_ey + head_radius * (cos_a * perp1[1] + sin_a * perp2[1])
            pz = shaft_ez + head_radius * (cos_a * perp1[2] + sin_a * perp2[2])
            glVertex3f(px, py, pz)
        glEnd()

        # === Draw cone base cap ===
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(shaft_ex, shaft_ey, shaft_ez)  # Center of base
        for i in range(segments, -1, -1):  # Reverse winding for correct face
            angle = 2.0 * math.pi * i / segments
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            px = shaft_ex + head_radius * (cos_a * perp1[0] + sin_a * perp2[0])
            py = shaft_ey + head_radius * (cos_a * perp1[1] + sin_a * perp2[1])
            pz = shaft_ez + head_radius * (cos_a * perp1[2] + sin_a * perp2[2])
            glVertex3f(px, py, pz)
        glEnd()

        # === Draw shaft end caps ===
        # Origin cap
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(ox, oy, oz)
        for i in range(segments, -1, -1):
            angle = 2.0 * math.pi * i / segments
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            px = ox + shaft_radius * (cos_a * perp1[0] + sin_a * perp2[0])
            py = oy + shaft_radius * (cos_a * perp1[1] + sin_a * perp2[1])
            pz = oz + shaft_radius * (cos_a * perp1[2] + sin_a * perp2[2])
            glVertex3f(px, py, pz)
        glEnd()

        glEnable(GL_DEPTH_TEST)

    def draw_vector(self, vector: 'Vector', color: RGBA = None) -> None:
        if color is None:
            color = vector._color
        ox, oy, oz = vector.origin[0], vector.origin[1], vector.origin[2]
        self.draw_arrow_3d(ox, oy, oz, ox + vector.x, oy + vector.y, oz + vector.z, color)

    def draw_animated_vector(self, animation: 'VectorAnimation') -> None:
        components, origin = animation.get_value()

        # Use dimmed opaque color instead of alpha to avoid color bleeding
        original_color = animation._start._color
        ghost_color = (original_color[0] * 0.4, original_color[1] * 0.4, original_color[2] * 0.4, 1.0)
        ox, oy, oz = animation._start.origin[0], animation._start.origin[1], animation._start.origin[2]
        self.draw_arrow_3d(ox, oy, oz,
                          ox + animation._start.x, oy + animation._start.y, oz + animation._start.z,
                          ghost_color, shaft_radius=0.02)

        ox, oy, oz = origin[0], origin[1], origin[2]
        self.draw_arrow_3d(ox, oy, oz,
                          ox + components[0], oy + components[1], oz + components[2],
                          animation._end._color)

    def draw_transformed_grid(self, animation: 'GridTransformAnimation') -> None:
        for start, end in animation.get_grid_points_3d():
            self.draw_line_3d(start[0], start[1], start[2], end[0], end[1], end[2], Colors.TRANSFORM_AFTER, 1.0)

    def draw_controls_hint(self, paused: bool) -> None:
        if not self._font:
            return

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self._width, self._height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        status = "PAUSED" if paused else "PLAYING"
        controls = "Space:Pause  R:Replay  Drag:Orbit  Shift+Drag:Pan  Scroll:Zoom  Esc:Exit"

        status_surface = self._font.render(status, True, (200, 200, 200) if not paused else (255, 200, 100))
        controls_surface = self._font_small.render(controls, True, (150, 150, 150))

        self._draw_text_surface(status_surface, 10, 10)
        self._draw_text_surface(controls_surface, 10, self._height - 25)

        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _draw_text_surface(self, surface: pygame.Surface, x: int, y: int) -> None:
        text_data = pygame.image.tostring(surface, "RGBA", True)
        w, h = surface.get_size()

        glEnable(GL_TEXTURE_2D)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 0); glVertex2f(x, y + h)
        glEnd()

        glDeleteTextures([texture])
        glDisable(GL_TEXTURE_2D)
