"""Timeline for sequencing animations."""

from __future__ import annotations
from typing import List, Dict
from linalg_viz.animation.animator import Animation


class TimelinePoint:
    """A point in time where animations can be added."""

    def __init__(self, timeline: 'Timeline', time: float):
        self._timeline = timeline
        self._time = time

    def add(self, obj) -> 'Timeline':
        # Handle Vector with pending animation
        if hasattr(obj, '_pending_animation') and obj._pending_animation is not None:
            self._timeline._add_at(self._time, obj._pending_animation)
        elif isinstance(obj, Animation):
            self._timeline._add_at(self._time, obj)
        return self._timeline


class Timeline:
    """Timeline for sequencing animations."""

    def __init__(self):
        self._animations: Dict[float, List[Animation]] = {}
        self._current_time = 0.0
        self._is_playing = False
        self._is_finished = False

    @property
    def duration(self) -> float:
        if not self._animations:
            return 0.0
        return max(t + max(a.duration for a in anims) for t, anims in self._animations.items())

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def is_finished(self) -> bool:
        return self._is_finished

    def at(self, time: float) -> TimelinePoint:
        return TimelinePoint(self, time)

    def _add_at(self, time: float, animation: Animation) -> None:
        if time not in self._animations:
            self._animations[time] = []
        self._animations[time].append(animation)

    def play(self) -> None:
        self._is_playing = True
        self._is_finished = False

    def pause(self) -> None:
        self._is_playing = False

    def stop(self) -> None:
        self._is_playing = False
        self._current_time = 0.0
        self._is_finished = False
        for anims in self._animations.values():
            for anim in anims:
                anim.reset()

    def update(self, dt: float) -> None:
        if not self._is_playing or self._is_finished:
            return

        self._current_time += dt
        all_finished = True

        for start_time, anims in self._animations.items():
            for anim in anims:
                if self._current_time >= start_time and not anim.is_finished:
                    anim.update(dt)
                if not anim.is_finished:
                    all_finished = False

        if all_finished and self._current_time >= self.duration:
            self._is_finished = True
            self._is_playing = False

    def get_active_animations(self) -> List[Animation]:
        return [a for t, anims in self._animations.items() for a in anims
                if self._current_time >= t and not a.is_finished]

    def get_all_animations(self) -> List[Animation]:
        return [a for anims in self._animations.values() for a in anims]
