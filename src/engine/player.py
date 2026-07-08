"""Player identity, roles, and the move-action vocabulary.

Boundary note (see PRD §{Ai}, §P): `Player.next_position` performs pure
coordinate arithmetic only. It does not consult a `Board`, so it will
happily "compute" a position that is out of bounds or sits on a barrier.
Rejecting/clamping such a move is deliberately left to the future
game_loop (which owns turn-scoped validation against both players' state
and the shared Board) — Player must not depend on Board, or a change to
grid geometry would ripple into player identity code that has nothing to
do with geometry.
"""

from __future__ import annotations

from enum import Enum

Coordinate = tuple[int, int]


class Action(Enum):
    """The 5-element action set `Ai` shared by both roles (PRD §{Ai}).

    STAY is a first-class action rather than "absence of a move" because
    the PRD calls out STAY as required for genuine wait/ambush/lure
    strategies — a Q-learning agent needs it as an explicit, equally
    selectable action, not a fallback behavior bolted on elsewhere.
    """

    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STAY = "STAY"


# (row_delta, col_delta) per action. Row increases downward, col increases
# rightward, matching the (row, col) tuple convention used across the
# engine package.
_DELTAS: dict[Action, Coordinate] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
}


class Role(Enum):
    """Cop and Thief are the same shape of entity with opposing objectives.

    A single `Player` class parameterized by `Role` is used instead of
    `Cop`/`Thief` subclasses because the two differ only in which reward
    row of `config.scoring` applies and in observation-radius perspective
    (PRD §R, §{Ωi}) — both are game_loop/reward concerns, not behavioral
    differences in how a player occupies or moves through the grid. A
    subclass split would duplicate identical move arithmetic for no gain.
    """

    COP = "COP"
    THIEF = "THIEF"


class Player:
    """A single agent's role and current position on the board."""

    def __init__(self, role: Role, start: Coordinate) -> None:
        self.role = role
        self.position = start

    def next_position(self, action: Action) -> Coordinate:
        """Return the coordinate `action` leads to, from the current position.

        Pure arithmetic, no bounds/barrier awareness — see module docstring
        for why that check is intentionally not performed here.
        """
        dr, dc = _DELTAS[action]
        r, c = self.position
        return (r + dr, c + dc)
