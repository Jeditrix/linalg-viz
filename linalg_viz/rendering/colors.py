"""Color definitions."""

from typing import Tuple

RGBA = Tuple[float, float, float, float]


class Colors:
    """Predefined colors (RGBA, 0-1 range)."""

    RED: RGBA = (0.9, 0.2, 0.2, 1.0)
    GREEN: RGBA = (0.2, 0.8, 0.3, 1.0)
    BLUE: RGBA = (0.2, 0.4, 0.9, 1.0)
    YELLOW: RGBA = (0.95, 0.85, 0.2, 1.0)
    CYAN: RGBA = (0.2, 0.8, 0.9, 1.0)
    MAGENTA: RGBA = (0.9, 0.2, 0.8, 1.0)
    ORANGE: RGBA = (1.0, 0.5, 0.1, 1.0)
    PURPLE: RGBA = (0.6, 0.3, 0.9, 1.0)
    WHITE: RGBA = (1.0, 1.0, 1.0, 1.0)
    GRAY: RGBA = (0.5, 0.5, 0.5, 1.0)

    BACKGROUND: RGBA = (0.12, 0.12, 0.15, 1.0)
    # Grid colors - darker values since they'll be rendered opaque
    # (originally had alpha < 1.0 which caused color bleeding in 3D)
    GRID: RGBA = (0.22, 0.22, 0.25, 1.0)
    GRID_MAJOR: RGBA = (0.30, 0.30, 0.35, 1.0)
    AXIS_X: RGBA = (0.8, 0.3, 0.3, 1.0)
    AXIS_Y: RGBA = (0.3, 0.8, 0.3, 1.0)
    AXIS_Z: RGBA = (0.3, 0.3, 0.8, 1.0)

    EIGEN_1: RGBA = (1.0, 0.8, 0.2, 1.0)
    EIGEN_2: RGBA = (0.2, 0.9, 0.8, 1.0)

    # Transform grid color - slightly dimmed since it was originally alpha=0.7
    TRANSFORM_AFTER: RGBA = (0.25, 0.5, 0.85, 1.0)

    _MAP = {
        "red": RED, "green": GREEN, "blue": BLUE, "yellow": YELLOW,
        "cyan": CYAN, "magenta": MAGENTA, "orange": ORANGE, "purple": PURPLE,
        "white": WHITE, "gray": GRAY, "grey": GRAY,
    }

    @classmethod
    def get(cls, name: str) -> RGBA:
        name = name.lower().replace("-", "_").replace(" ", "_")
        if name in cls._MAP:
            return cls._MAP[name]
        raise ValueError(f"Unknown color: {name}")

    @classmethod
    def with_alpha(cls, color: RGBA, alpha: float) -> RGBA:
        return (color[0], color[1], color[2], alpha)

    @classmethod
    def lerp(cls, c1: RGBA, c2: RGBA, t: float) -> RGBA:
        return tuple(a + (b - a) * t for a, b in zip(c1, c2))
