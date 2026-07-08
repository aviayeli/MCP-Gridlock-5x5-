"""Grid board model: bounds checking and randomized barrier placement.

Coordinates are plain `tuple[int, int]` (row, col) rather than a small
`Coordinate` class. Tuples are immutable and hashable out of the box, which
matters because both barrier membership (a `set`) and, later, Q-table
observation keys (per PRD §{Ωi}) are indexed by position — a mutable
coordinate object would require hand-rolled `__hash__`/`__eq__` for no
behavioral benefit here.
"""

from __future__ import annotations

import random

Coordinate = tuple[int, int]


class Board:
    """Fixed-size grid that owns barrier layout and in-bounds checks.

    The Board is deliberately ignorant of players, moves, or game rules
    beyond static geometry (bounds + barriers). Legality of a *move* (e.g.
    "can the Cop step from A to B this turn") is a game_loop concern, since
    that also depends on turn-scoped state such as the opponent's position;
    Board only answers the position-scoped question "is this single cell a
    barrier / in bounds".
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        barrier_count: int,
        exclude: set[Coordinate] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Build a board and place `barrier_count` barriers immediately.

        `rng` accepts an injected `random.Random` (rather than reseeding the
        global `random` module) so a caller can pin a seed per the PRD's
        "Determinism & Reproducibility" requirement (PLAN.md §1.5) —
        replaying a game with the same seed must yield the same barrier
        layout, which is impossible if placement silently depends on global
        random state shared with unrelated code.
        """
        self.rows = rows
        self.cols = cols
        self._rng = rng if rng is not None else random.Random()
        self._barriers: set[Coordinate] = set()
        self._place_barriers(barrier_count, exclude or set())

    def _place_barriers(self, count: int, exclude: set[Coordinate]) -> None:
        """Sample `count` distinct, non-excluded cells as barriers.

        Excluding start cells (typically the Cop's and Thief's initial
        positions) is required because a barrier placed under a player at
        episode start would make that player's own cell impassable to
        re-enter after a STAY/round-trip, an unintended and unrecoverable
        soft-lock that has nothing to do with the intended difficulty of
        navigating *around* barriers.
        """
        candidates = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in exclude
        ]
        count = min(count, len(candidates))
        self._barriers = set(self._rng.sample(candidates, count))

    def in_bounds(self, coord: Coordinate) -> bool:
        """Return whether `coord` lies within the grid's row/col extent."""
        r, c = coord
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_barrier(self, coord: Coordinate) -> bool:
        """Return whether `coord` is an impassable barrier cell."""
        return coord in self._barriers

    def barrier_coordinates(self) -> frozenset[Coordinate]:
        """Return an immutable snapshot of all barrier cells.

        A frozenset copy is returned (rather than the live `set`) so callers
        — e.g. an observation payload handed to an LLM agent — cannot
        mutate board state through a reference they were only meant to
        read.
        """
        return frozenset(self._barriers)
