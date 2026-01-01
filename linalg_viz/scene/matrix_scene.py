"""Scene for visualizing matrix/vector arithmetic with numbers."""

from __future__ import annotations
from typing import Optional, List
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *

from linalg_viz.rendering.matrix_display import MatrixDisplay
from linalg_viz.rendering.colors import Colors


class MatrixScene:
    """Scene for displaying matrix operations with actual numbers."""

    def __init__(self, width: int = 1000, height: int = 600, title: str = "Matrix Visualization"):
        self._width = width
        self._height = height
        self._title = title
        self._display = MatrixDisplay(width, height)
        self._running = False
        self._clock = None
        self._screen = None

        # Animation state
        self._animation_step = 0
        self._animation_timer = 0.0
        self._step_duration = 1.0  # seconds per step
        self._paused = False

    def _init_pygame(self) -> None:
        pygame.init()
        pygame.display.set_caption(self._title)
        self._screen = pygame.display.set_mode((self._width, self._height), DOUBLEBUF | OPENGL)
        self._clock = pygame.time.Clock()

        # Setup 2D orthographic projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self._width, self._height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._display.init()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == QUIT:
                self._running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self._running = False
                elif event.key == K_SPACE:
                    self._paused = not self._paused
                elif event.key == K_r:
                    self._animation_step = 0
                    self._animation_timer = 0.0
                elif event.key == K_RIGHT:
                    self._animation_step += 1
                elif event.key == K_LEFT:
                    self._animation_step = max(0, self._animation_step - 1)

    def show_matrix_vector_multiply(self, matrix: np.ndarray, vector: np.ndarray) -> None:
        """Animate matrix-vector multiplication step by step."""
        self._init_pygame()
        self._running = True

        rows, cols = matrix.shape
        result = matrix @ vector
        total_steps = rows + 1  # One step per row + final result

        while self._running:
            dt = self._clock.tick(60) / 1000.0
            self._handle_events()

            if not self._paused:
                self._animation_timer += dt
                if self._animation_timer >= self._step_duration:
                    self._animation_timer = 0.0
                    if self._animation_step < total_steps:
                        self._animation_step += 1

            # Clear
            glClearColor(*Colors.BACKGROUND)
            glClear(GL_COLOR_BUFFER_BIT)

            # Current highlighted row
            current_row = min(self._animation_step, rows - 1)

            # Layout
            start_x = 50
            start_y = 100
            spacing = 40

            # Draw title
            self._display._draw_text("Matrix × Vector = Result", start_x, 30, (200, 200, 200),
                                     self._display._font_large)

            # Draw matrix with highlighted row
            mw, mh = self._display.draw_matrix(matrix, start_x, start_y,
                                                highlight_row=current_row if self._animation_step < rows else -1)

            # Draw multiply sign
            x = start_x + mw + spacing
            self._display.draw_multiply(x, start_y, mh)

            # Draw vector with highlighted element
            x += spacing + 20
            vw, vh = self._display.draw_vector(vector, x, start_y,
                                               highlight_idx=current_row if self._animation_step < rows else -1)

            # Draw equals
            x += vw + spacing
            self._display.draw_equals(x, start_y, mh)

            # Draw result vector (progressively revealed)
            x += spacing + 20
            result_partial = np.zeros_like(result)
            for i in range(min(self._animation_step, rows)):
                result_partial[i] = result[i]

            if self._animation_step > 0:
                self._display.draw_vector(result_partial, x, start_y, color=(100, 255, 100),
                                          highlight_idx=current_row - 1 if self._animation_step <= rows else -1)

            # Show current calculation
            if self._animation_step > 0 and self._animation_step <= rows:
                row_idx = self._animation_step - 1
                calc_y = start_y + mh + 60

                # Build calculation string with clean formatting
                terms = []
                for j in range(cols):
                    m_val = self._display._format_number(matrix[row_idx, j])
                    v_val = self._display._format_number(vector[j])
                    terms.append(f"{m_val}×{v_val}")
                result_val = self._display._format_number(result[row_idx])
                calc_str = f"Row {row_idx + 1}: " + " + ".join(terms) + f" = {result_val}"
                self._display._draw_text(calc_str, start_x, calc_y, (255, 255, 100))

            # Draw controls hint
            controls = "Space:Pause  R:Restart  ←→:Step  Esc:Exit"
            self._display._draw_text(controls, 10, self._height - 30, (150, 150, 150))

            pygame.display.flip()

        pygame.quit()

    def show_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> None:
        """Animate matrix-matrix multiplication step by step."""
        self._init_pygame()
        self._running = True

        rows_a, cols_a = A.shape
        rows_b, cols_b = B.shape
        result = A @ B
        total_steps = rows_a * cols_b + 1

        while self._running:
            dt = self._clock.tick(60) / 1000.0
            self._handle_events()

            if not self._paused:
                self._animation_timer += dt
                if self._animation_timer >= self._step_duration:
                    self._animation_timer = 0.0
                    if self._animation_step < total_steps:
                        self._animation_step += 1

            # Clear
            glClearColor(*Colors.BACKGROUND)
            glClear(GL_COLOR_BUFFER_BIT)

            # Current position in result matrix
            current_idx = min(self._animation_step, rows_a * cols_b - 1)
            current_row = current_idx // cols_b
            current_col = current_idx % cols_b

            # Layout
            start_x = 30
            start_y = 100
            spacing = 30

            # Draw title
            self._display._draw_text("Matrix × Matrix = Result", start_x, 30, (200, 200, 200),
                                     self._display._font_large)

            # Draw matrix A with highlighted row
            highlight_r = current_row if self._animation_step < total_steps - 1 else -1
            mw_a, mh_a = self._display.draw_matrix(A, start_x, start_y, highlight_row=highlight_r)

            # Draw multiply sign
            x = start_x + mw_a + spacing
            self._display.draw_multiply(x, start_y, mh_a)

            # Draw matrix B with highlighted column
            x += spacing + 10
            highlight_c = current_col if self._animation_step < total_steps - 1 else -1
            mw_b, mh_b = self._display.draw_matrix(B, x, start_y, highlight_col=highlight_c)

            # Draw equals
            x += mw_b + spacing
            self._display.draw_equals(x, start_y, mh_a)

            # Draw result matrix (progressively revealed)
            x += spacing + 10
            result_partial = np.zeros_like(result)
            for idx in range(min(self._animation_step, rows_a * cols_b)):
                r, c = idx // cols_b, idx % cols_b
                result_partial[r, c] = result[r, c]

            if self._animation_step > 0:
                prev_row = (current_idx - 1) // cols_b if current_idx > 0 else -1
                prev_col = (current_idx - 1) % cols_b if current_idx > 0 else -1
                self._display.draw_matrix(result_partial, x, start_y, color=(100, 255, 100),
                                          highlight_row=prev_row if self._animation_step < total_steps else -1,
                                          highlight_col=prev_col if self._animation_step < total_steps else -1)

            # Show current calculation
            if self._animation_step > 0 and self._animation_step < total_steps:
                calc_y = start_y + max(mh_a, mh_b) + 60
                prev_idx = self._animation_step - 1
                r, c = prev_idx // cols_b, prev_idx % cols_b

                # Build calculation string with clean formatting
                terms = []
                for k in range(cols_a):
                    a_val = self._display._format_number(A[r, k])
                    b_val = self._display._format_number(B[k, c])
                    terms.append(f"{a_val}×{b_val}")
                result_val = self._display._format_number(result[r, c])
                calc_str = f"C[{r+1},{c+1}]: " + " + ".join(terms) + f" = {result_val}"
                self._display._draw_text(calc_str, start_x, calc_y, (255, 255, 100))

            # Draw controls hint
            controls = "Space:Pause  R:Restart  ←→:Step  Esc:Exit"
            self._display._draw_text(controls, 10, self._height - 30, (150, 150, 150))

            pygame.display.flip()

        pygame.quit()

    def show_dot_product(self, a: np.ndarray, b: np.ndarray) -> None:
        """Animate dot product calculation step by step."""
        self._init_pygame()
        self._running = True

        n = len(a)
        result = np.dot(a, b)
        total_steps = n + 1

        while self._running:
            dt = self._clock.tick(60) / 1000.0
            self._handle_events()

            if not self._paused:
                self._animation_timer += dt
                if self._animation_timer >= self._step_duration:
                    self._animation_timer = 0.0
                    if self._animation_step < total_steps:
                        self._animation_step += 1

            # Clear
            glClearColor(*Colors.BACKGROUND)
            glClear(GL_COLOR_BUFFER_BIT)

            current_idx = min(self._animation_step, n - 1)

            # Layout
            start_x = 100
            start_y = 150
            spacing = 40

            # Draw title
            self._display._draw_text("Vector Dot Product: a · b", start_x, 50, (200, 200, 200),
                                     self._display._font_large)

            # Draw vector a (as row)
            self._display._draw_text("a =", start_x, start_y + 10, (255, 150, 150))
            x = start_x + 50
            aw, ah = self._display.draw_vector(a, x, start_y - 20,
                                               color=(255, 150, 150),
                                               highlight_idx=current_idx if self._animation_step < n else -1)

            # Draw dot
            x += aw + spacing
            self._display._draw_text("·", x, start_y + 10, (200, 200, 200), self._display._font_large)

            # Draw vector b
            x += spacing + 20
            self._display._draw_text("b =", x, start_y + 10, (150, 150, 255))
            x += 50
            bw, bh = self._display.draw_vector(b, x, start_y - 20,
                                               color=(150, 150, 255),
                                               highlight_idx=current_idx if self._animation_step < n else -1)

            # Draw equals and result
            x += bw + spacing
            self._display.draw_equals(x, start_y - 20, ah)

            x += spacing + 20
            if self._animation_step >= n:
                result_text = self._display._format_number(result)
                self._display._draw_text(result_text, x, start_y + 10, (100, 255, 100),
                                         self._display._font_large)

            # Show current calculation
            calc_y = start_y + ah + 80
            if self._animation_step > 0:
                terms = []
                running_sum = 0.0
                for i in range(min(self._animation_step, n)):
                    a_val = self._display._format_number(a[i])
                    b_val = self._display._format_number(b[i])
                    terms.append(f"{a_val}×{b_val}")
                    running_sum += a[i] * b[i]

                result_val = self._display._format_number(running_sum)
                calc_str = " + ".join(terms) + f" = {result_val}"
                self._display._draw_text(calc_str, start_x, calc_y, (255, 255, 100))

            # Draw controls hint
            controls = "Space:Pause  R:Restart  ←→:Step  Esc:Exit"
            self._display._draw_text(controls, 10, self._height - 30, (150, 150, 150))

            pygame.display.flip()

        pygame.quit()
