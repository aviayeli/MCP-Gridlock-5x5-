"""The role-swap schedule: which `Team` plays Cop in a given game index.

Split out from tournament.py purely to respect this package's 150-line-per
-file cap. Kept as its own module rather than folded into
tournament_report.py because it is a scheduling *policy* (a pure function
of game index), not part of the report schema itself — tournament_report.py
stays focused on "what does a finished game/series look like," while this
module answers "who plays which role, before any game is even played."
"""

from __future__ import annotations

from engine.tournament_report import Team


def cop_team_for_game(game_index: int) -> Team:
    """Return which Team plays Cop in game `game_index` (0-based).

    Team Alpha plays Cop on even-indexed games (0, 2, 4, ...) and Thief on
    odd-indexed games; Team Beta always takes the opposite role. Swapping
    every single game (rather than e.g. splitting the series into a first
    -half/second-half block) spreads any transient, game-index-correlated
    factor (such as an under-trained Q-table early in the series) evenly
    across both teams instead of concentrating it on whichever team holds
    a given role during the early games. For `num_games=6` (the current
    config), this gives each team exactly 3 Cop games and 3 Thief games,
    so neither team retains a permanent positional advantage.
    """
    return Team.ALPHA if game_index % 2 == 0 else Team.BETA
