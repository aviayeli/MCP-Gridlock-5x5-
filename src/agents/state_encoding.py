"""Fog-of-war observation -> discrete Q-table state key (PRD §{Ωi}, §S).

Split out from q_agent.py purely to respect this package's 150-line-per-file
cap; "how do we compress a wire-format observation into something a Q-table
can key on" is also a self-contained concern, separate from action selection
and hyperparameters.

WHY relative encoding, not absolute positions
----------------------------------------------
The PRD's own state space `S` (§S) is `25 (cop) x 25 (thief) x C(25,<=5)
(barriers) x 26 (move_count)` — already large, and that is the *true*
game state, not even a single agent's Q-table key. If a Q-table were keyed
on the raw `self_position`/`opponent_position` pairs straight out of
`GameTools.get_observation`, two facts would follow, both bad:

1. State-space blowup: absolute self x opponent position alone is already
   25 x 26 (25 opponent cells + "unseen") = 650 keys, before barriers are
   even considered, for a game where the *tactically relevant* fact is
   almost always "where is the opponent relative to me", not "what are our
   two absolute grid cells".
2. No generalization: a Q-value learned for "opponent one cell to my right
   at position (0,0)" would not transfer to the geometrically identical
   situation "opponent one cell to my right at position (3,3)" — the agent
   would have to relearn the same tactic at every position on the board
   independently, which defeats the point of tabular Q-learning converging
   in a small number of episodes (`num_games = 6`).

Encoding the opponent's position as a **relative delta from the agent's own
position** collapses all board-position-symmetric situations onto the same
state key, so a tactic learned in one corner of the grid immediately
transfers to the equivalent situation anywhere else on the board.

WHY `None` sentinel for fog-of-war, not an out-of-range delta
---------------------------------------------------------------
When the opponent is outside `VISIBILITY_RADIUS`, `GameTools.get_observation`
reports `opponent_position: None` (see `engine.observation.UNSEEN`). This
module preserves that same `None` sentinel in the relative-delta slot of the
state key, rather than inventing some numeric "unseen" delta, for two
reasons: (a) it can never collide with a real relative delta (every visible
delta is a small `(int, int)` tuple, and `None` is not a tuple), and (b) it
keeps the "unseen" concept vocabulary consistent with the rest of the
codebase (`engine.observation.UNSEEN is None`). Fog-of-war is deliberately
kept as its own distinguishable state (not folded into, say, "opponent very
far away") because the whole point of a Dec-POMDP over a Dec-MDP (PRD §{Ωi})
is that *not knowing* where the opponent is should be learnable as its own
regime — e.g. a Thief might learn to patrol/idle differently under
uncertainty than when it can see exactly where the Cop is.

WHY a 4-bit adjacency mask for barriers, not full relative barrier positions
------------------------------------------------------------------------------
Encoding every barrier's position relative to the agent (mirroring the
opponent-delta treatment) would multiply the state space by the number of
distinct barrier layouts visible from a given relative frame -- combinatorial,
and mostly irrelevant, since barriers only matter to a single turn's decision
insofar as they block an adjacent move right now. This module instead encodes
only "is a barrier immediately adjacent in each of the 4 movement directions"
as a 4-bit mask (16 possible values) taken from `self_position`. This is a
deliberate lossy simplification: a barrier two cells away in a direction the
agent isn't about to step into carries no information a 1-step-lookahead
Q-agent can act on this turn, whereas an adjacent barrier directly removes a
legal move from consideration. Combined with the opponent-delta encoding
(bounded to Manhattan distance <= `VISIBILITY_RADIUS` = 2, i.e. 13 possible
deltas, + 1 for unseen = 14 possibilities), the full state space is a tiny
`14 * 16 = 224` keys -- small enough to explore thoroughly within
`num_games = 6` episodes, in sharp contrast to the combinatorial blowup an
absolute or full-barrier-relative encoding would produce.
"""

from __future__ import annotations

Coordinate = tuple[int, int]

# One state-key slot: the opponent's position relative to the agent's own
# position (row_delta, col_delta), or `None` if fog-of-war hides them this
# turn. See module docstring for why `None` (not a numeric sentinel) is used.
RelativeOpponent = tuple[int, int] | None

# The full, hashable, deterministic Q-table key for one observation.
StateKey = tuple[RelativeOpponent, int]

# (row_delta, col_delta) checked for each bit of the barrier-adjacency mask,
# in the same UP/DOWN/LEFT/RIGHT order as `engine.player.Action` (STAY has
# no associated direction, so it is not part of the mask).
_ADJACENT_DELTAS: tuple[Coordinate, ...] = (
    (-1, 0),  # UP
    (1, 0),  # DOWN
    (0, -1),  # LEFT
    (0, 1),  # RIGHT
)


def _relative_opponent(
    self_position: Coordinate, opponent_position: list[int] | None
) -> RelativeOpponent:
    """Return the opponent's position relative to `self_position`, or None."""
    if opponent_position is None:
        return None
    return (
        opponent_position[0] - self_position[0],
        opponent_position[1] - self_position[1],
    )


def _barrier_adjacency_mask(
    self_position: Coordinate, barriers: list[list[int]]
) -> int:
    """Return a 4-bit mask: is a barrier adjacent in each UP/DOWN/LEFT/RIGHT."""
    barrier_cells = {tuple(cell) for cell in barriers}
    mask = 0
    for bit, (dr, dc) in enumerate(_ADJACENT_DELTAS):
        neighbor = (self_position[0] + dr, self_position[1] + dc)
        if neighbor in barrier_cells:
            mask |= 1 << bit
    return mask


def encode_observation(observation: dict) -> StateKey:
    """Convert a wire-format observation dict into a discrete Q-table key.

    `observation` is the exact JSON shape returned by
    `GameTools.get_observation` (`self_position`/`opponent_position` as
    lists or None, `barriers` as a list of `[row, col]` lists, `move_count`
    as an int) -- this is what a real MCP-driven agent actually receives,
    so encoding is done directly off that shape rather than the engine's
    internal `Observation` dataclass.

    `move_count` is deliberately excluded from the key: it is a monotonic
    game-clock signal (PRD §{Ωi} calls it "freely-visible", not tactical
    information), and folding its 26 possible values into the key would
    multiply the already-tiny state space >20x for a dimension that a
    1-step Q-lookup can't act on any differently turn-to-turn.

    Pure and deterministic: the same `observation` dict always yields the
    same `StateKey`, and no mutable/global state is consulted.
    """
    self_position = tuple(observation["self_position"])
    opponent_relative = _relative_opponent(
        self_position, observation["opponent_position"]
    )
    barrier_mask = _barrier_adjacency_mask(self_position, observation["barriers"])
    return (opponent_relative, barrier_mask)
