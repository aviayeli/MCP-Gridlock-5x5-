"""Fog-of-war observation derivation (PRD §{Ωi}, §O).

Split out from game_loop.py to respect this package's 150-line-per-file
cap; "what can agent i currently see" is also a self-contained concern
separate from turn resolution and reward assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.board import Coordinate

# Manhattan-distance visibility radius: how close the opponent must be
# before their exact cell is revealed. This is a game_loop-level design
# constant, distinct from config/game_config.json (whose schema has no
# visibility field). Chosen as 2 because on a 5x5 board (max Manhattan
# distance 8) it keeps most of the board genuinely fogged while still
# giving a closing Cop at least one confirmed sighting before contact —
# radius 1 would make "unseen" the overwhelmingly common case even at
# close range, and radius >=4 would erase partial observability almost
# entirely on a board this small.
VISIBILITY_RADIUS = 2

# Sentinel used in place of a coordinate when the opponent is out of range.
# `None` is used (rather than e.g. (-1, -1)) so "unseen" can never be
# mistaken for a real, if invalid-looking, board coordinate.
UNSEEN: Coordinate | None = None


@dataclass(frozen=True)
class Observation:
    """One agent's partial view `ωi` of the post-transition state."""

    self_position: Coordinate
    opponent_position: Coordinate | None
    barriers: frozenset[Coordinate]
    move_count: int


def manhattan_distance(a: Coordinate, b: Coordinate) -> int:
    """L1 grid distance, used for the visibility-radius check."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_observation(
    self_position: Coordinate,
    opponent_position: Coordinate,
    barriers: frozenset[Coordinate],
    move_count: int,
    radius: int = VISIBILITY_RADIUS,
) -> Observation:
    """Derive one agent's `ωi` (PRD §O).

    Deterministic, not noisy: reveals the opponent's exact cell iff their
    Manhattan distance from `self_position` is within `radius`, otherwise
    reports UNSEEN. Own position, the full barrier layout, and move_count
    are always visible, per the PRD's Ωi definition — barriers are static
    "public terrain" and move_count is a shared clock signal, not tactical
    information, so neither is subject to fog of war.
    """
    visible = manhattan_distance(self_position, opponent_position) <= radius
    return Observation(
        self_position=self_position,
        opponent_position=opponent_position if visible else UNSEEN,
        barriers=barriers,
        move_count=move_count,
    )
