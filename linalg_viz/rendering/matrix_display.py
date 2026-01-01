"""Matrix and vector numeric display renderer.

Displays matrices and vectors as actual numbers with brackets,
and animates multiplication step-by-step.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import pygame
from OpenGL.GL import *
import numpy as np


class MatrixDisplay:
    """Renders matrices and vectors as numeric displays."""

    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self._font = None
        self._font_large = None
        self._cell_width = 50
        self._cell_height = 36
        self._bracket_width = 12

    def _format_number(self, val: float) -> str:
        """Format number cleanly - no decimals if integer."""
        if val == int(val):
            return str(int(val))
        elif abs(val) < 0.01:
            return "0"
        elif abs(val - round(val, 1)) < 0.01:
            return f"{val:.1f}"
        else:
            return f"{val:.2f}"

    def init(self) -> None:
        pygame.font.init()
        self._font = pygame.font.SysFont('monospace', 24)
        self._font_large = pygame.font.SysFont('monospace', 32)

    def _draw_text(self, text: str, x: int, y: int, color: Tuple[int, int, int] = (255, 255, 255),
                   font: pygame.font.Font = None) -> Tuple[int, int]:
        """Draw text and return its size."""
        if font is None:
            font = self._font
        surface = font.render(text, True, color)
        text_data = pygame.image.tostring(surface, "RGBA", True)
        w, h = surface.get_size()

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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
        glDisable(GL_BLEND)

        return w, h

    def _draw_bracket(self, x: int, y: int, height: int, left: bool = True,
                      color: Tuple[int, int, int] = (200, 200, 200)) -> None:
        """Draw a matrix bracket."""
        glDisable(GL_TEXTURE_2D)
        glColor3f(color[0]/255, color[1]/255, color[2]/255)
        glLineWidth(2.0)

        w = self._bracket_width
        glBegin(GL_LINE_STRIP)
        if left:
            glVertex2f(x + w, y)
            glVertex2f(x, y)
            glVertex2f(x, y + height)
            glVertex2f(x + w, y + height)
        else:
            glVertex2f(x, y)
            glVertex2f(x + w, y)
            glVertex2f(x + w, y + height)
            glVertex2f(x, y + height)
        glEnd()

    def draw_matrix(self, data: np.ndarray, x: int, y: int,
                    color: Tuple[int, int, int] = (255, 255, 255),
                    highlight_row: int = -1, highlight_col: int = -1,
                    highlight_color: Tuple[int, int, int] = (255, 255, 100)) -> Tuple[int, int]:
        """Draw a matrix with brackets. Returns (width, height)."""
        rows, cols = data.shape
        total_height = rows * self._cell_height
        total_width = cols * self._cell_width + 2 * self._bracket_width

        # Draw brackets
        self._draw_bracket(x, y, total_height, left=True, color=color)
        self._draw_bracket(x + total_width - self._bracket_width, y, total_height, left=False, color=color)

        # Draw numbers
        for i in range(rows):
            for j in range(cols):
                text = self._format_number(data[i, j])

                # Determine color
                cell_color = color
                if i == highlight_row or j == highlight_col:
                    cell_color = highlight_color
                if i == highlight_row and j == highlight_col:
                    cell_color = (100, 255, 100)

                # Center text in cell
                cx = x + self._bracket_width + j * self._cell_width + (self._cell_width - len(text) * 10) // 2
                cy = y + i * self._cell_height + 6
                self._draw_text(text, cx, cy, cell_color)

        return total_width, total_height

    def draw_vector(self, data: np.ndarray, x: int, y: int,
                    color: Tuple[int, int, int] = (255, 255, 255),
                    highlight_idx: int = -1,
                    highlight_color: Tuple[int, int, int] = (255, 255, 100)) -> Tuple[int, int]:
        """Draw a column vector with brackets. Returns (width, height)."""
        n = len(data)
        total_height = n * self._cell_height
        total_width = self._cell_width + 2 * self._bracket_width

        # Draw brackets
        self._draw_bracket(x, y, total_height, left=True, color=color)
        self._draw_bracket(x + total_width - self._bracket_width, y, total_height, left=False, color=color)

        # Draw numbers
        for i in range(n):
            text = self._format_number(data[i])
            cell_color = highlight_color if i == highlight_idx else color

            # Center text in cell
            cx = x + self._bracket_width + (self._cell_width - len(text) * 10) // 2
            cy = y + i * self._cell_height + 6
            self._draw_text(text, cx, cy, cell_color)

        return total_width, total_height

    def draw_equals(self, x: int, y: int, height: int) -> int:
        """Draw an equals sign centered vertically. Returns width."""
        cy = y + height // 2 - 15
        self._draw_text("=", x, cy, (200, 200, 200), self._font_large)
        return 30

    def draw_multiply(self, x: int, y: int, height: int) -> int:
        """Draw a multiplication dot centered vertically. Returns width."""
        cy = y + height // 2 - 15
        self._draw_text("×", x, cy, (200, 200, 200), self._font_large)
        return 30

    def draw_scalar(self, value: float, x: int, y: int, height: int,
                    color: Tuple[int, int, int] = (100, 255, 100)) -> int:
        """Draw a scalar result. Returns width."""
        cy = y + height // 2 - 15
        text = f"{value:.2f}"
        self._draw_text(text, x, cy, color, self._font_large)
        return len(text) * 15
