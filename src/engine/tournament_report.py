"""JSON-serializable tournament report schema and game-record construction.

Split out from tournament.py purely to respect this package's 150-line-per
-file cap; "what does one game's record look like" and "how do we roll
records into a final series report" are also a self-contained concern,
separate from the game-driving/role-swap logic tournament.py owns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from engine.config_loader import GameConfig
from engine.game_loop import TurnResult


class Team(Enum):
    """The two competing teams, independent of which Role either currently plays.

    Kept in this module (rather than tournament.py) because Team identity is
    as much a report-schema concern (cop_team/thief_team/final_scores/winner
    fields) as a scheduling one; tournament.py imports it from here, which
    also avoids a circular import between the two modules.
    """

    ALPHA = "ALPHA"
    BETA = "BETA"


class Outcome(Enum):
    """How a game ended, per `GameLoop.step`'s terminal conditions."""

    CAPTURE = "CAPTURE"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class GameRecord:
    """One game's result, structured for both in-process use and JSON export."""

    game_index: int
    cop_team: Team
    thief_team: Team
    moves: int
    cop_reward: int
    thief_reward: int
    outcome: Outcome

    def to_dict(self) -> dict:
        """Render as a JSON-serializable dict (Enums rendered as `.value`)."""
        return {
            "game_index": self.game_index,
            "cop_team": self.cop_team.value,
            "thief_team": self.thief_team.value,
            "moves": self.moves,
            "cop_reward": self.cop_reward,
            "thief_reward": self.thief_reward,
            "outcome": self.outcome.value,
        }


def build_game_record(
    game_index: int,
    cop_team: Team,
    thief_team: Team,
    result: TurnResult,
    config: GameConfig,
) -> GameRecord:
    """Construct a `GameRecord` from one finished game's terminal `TurnResult`.

    Outcome is inferred from which scoring payout the Cop actually received,
    rather than threading a separate `captured: bool` out of `GameLoop`
    (which exposes no such flag), because `cop_reward == scoring.cop_win`
    and `cop_reward == scoring.cop_loss` are already the two mutually
    exclusive, exhaustive terminal outcomes defined by `GameLoop.step`.
    """
    outcome = (
        Outcome.CAPTURE
        if result.cop_reward == config.scoring.cop_win
        else Outcome.TIMEOUT
    )
    return GameRecord(
        game_index=game_index,
        cop_team=cop_team,
        thief_team=thief_team,
        moves=result.move_count,
        cop_reward=result.cop_reward,
        thief_reward=result.thief_reward,
        outcome=outcome,
    )


def _winner(team_scores: dict[Team, int]) -> str:
    """Return the higher-cumulative-score team's name, or `"tie"`.

    A tie is treated as a first-class, valid series outcome rather than
    resolved by an arbitrary tiebreak (e.g. "earlier team wins"), because
    the role-swap schedule is specifically designed to make the series
    fair-by-construction — a symmetric result across a symmetric schedule
    is a legitimate finding to report, not a corner case to paper over.
    """
    alpha, beta = team_scores[Team.ALPHA], team_scores[Team.BETA]
    if alpha == beta:
        return "tie"
    return Team.ALPHA.value if alpha > beta else Team.BETA.value


def build_report(
    config: GameConfig,
    records: list[GameRecord],
    team_scores: dict[Team, int],
) -> dict:
    """Assemble the final series report as a plain JSON-serializable dict.

    Schema (top-level keys):
      - `num_games`: expected series length, read from config (never
        hardcoded), so the report is self-describing even if the config
        changes between runs.
      - `games`: one entry per `GameRecord.to_dict()`, in play order —
        game index, each role's team, moves taken, terminal rewards, and
        capture-vs-timeout outcome.
      - `final_scores`: cumulative score per team, keyed by team name.
      - `winner`: the higher-cumulative-score team's name, or `"tie"`.
    This flat shape (rather than nesting games under teams, or vice versa)
    was chosen because per-game detail and per-team totals are independent
    projections of the same underlying data — a flat report lets a
    consumer read either projection directly, without reconstructing one
    from the other.
    """
    return {
        "num_games": config.num_games,
        "games": [record.to_dict() for record in records],
        "final_scores": {team.value: score for team, score in team_scores.items()},
        "winner": _winner(team_scores),
    }
