"""Turn resolution, terminal conditions, and reward assignment.

Composes a `Board` and two `Player`s (one COP, one THIEF) rather than
reimplementing grid geometry or move arithmetic — see board.py's and
player.py's module docstrings for why bounds/barrier checks and position
arithmetic deliberately live there instead of here. This module owns only
the turn-scoped concerns that both of those files explicitly leave open:
move legality resolution, terminal-condition checks, and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.board import Board, Coordinate
from engine.config_loader import GameConfig, load_config
from engine.observation import Observation, build_observation
from engine.player import Action, Player, Role


@dataclass(frozen=True)
class TurnResult:
    """Outcome of a single resolved turn, returned to both calling agents."""

    cop_observation: Observation
    thief_observation: Observation
    move_count: int
    done: bool
    cop_reward: int | None
    thief_reward: int | None


class GameLoop:
    """Drives one episode: turn resolution, terminal checks, scoring.

    Owns exactly one `Board` and two `Player`s. Running `num_games` (config)
    episodes is a caller-level concern (e.g. an experiment runner that
    instantiates a fresh `GameLoop` per game) — this class deliberately
    knows nothing about that outer loop, matching the layering already
    established by Board/Player (each layer owns only what the layer above
    it cannot reasonably own).
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        cop_start: Coordinate = (0, 0),
        thief_start: Coordinate = (4, 4),
    ) -> None:
        """Set up the board and both players for one episode."""
        self.config = config or load_config()
        self.board = Board(
            self.config.rows,
            self.config.cols,
            self.config.max_barriers,
            exclude={cop_start, thief_start},
        )
        self.cop = Player(Role.COP, cop_start)
        self.thief = Player(Role.THIEF, thief_start)
        self.move_count = 0
        self.done = False

    def _resolve_move(self, player: Player, action: Action) -> Coordinate:
        """Validate a candidate move; an illegal move resolves to STAY.

        An out-of-bounds or barrier-blocked destination collapses to the
        player's current position rather than raising or being rejected
        here, because `game_loop` has no channel back to the calling agent
        mid-turn — rejection/re-prompting is explicitly the MCP server's
        job, one layer above this engine (PRD §{Ai}). Resolving to STAY
        keeps the deterministic transition function `P` total: every
        (state, action) pair yields a next state, never an error, which is
        required for the reproducibility guarantee in PRD §5.
        """
        candidate = player.next_position(action)
        legal = self.board.in_bounds(candidate) and not self.board.is_barrier(
            candidate
        )
        return candidate if legal else player.position

    def step(self, cop_action: Action, thief_action: Action) -> TurnResult:
        """Resolve one simultaneous turn and return both agents' results.

        Terminal condition ordering: capture is checked before the
        `max_moves` timeout. Both conditions can only be evaluated after
        `move_count` is incremented, and if a capture happens to land on
        the final allowed move, it must still score as a Cop win rather
        than being shadowed by a simultaneous timeout — capture is the
        game's substantive win condition; timeout is only a fallback for
        the case where neither agent ever collides.
        """
        if self.done:
            raise RuntimeError("step() called after episode already ended")

        self.cop.position = self._resolve_move(self.cop, cop_action)
        self.thief.position = self._resolve_move(self.thief, thief_action)
        self.move_count += 1

        captured = self.cop.position == self.thief.position
        timed_out = not captured and self.move_count >= self.config.max_moves
        self.done = captured or timed_out

        cop_reward = thief_reward = None
        scoring = self.config.scoring
        if captured:
            cop_reward, thief_reward = scoring.cop_win, scoring.thief_loss
        elif timed_out:
            cop_reward, thief_reward = scoring.cop_loss, scoring.thief_win

        barriers = self.board.barrier_coordinates()
        cop_obs = build_observation(
            self.cop.position, self.thief.position, barriers, self.move_count
        )
        thief_obs = build_observation(
            self.thief.position, self.cop.position, barriers, self.move_count
        )
        return TurnResult(
            cop_observation=cop_obs,
            thief_observation=thief_obs,
            move_count=self.move_count,
            done=self.done,
            cop_reward=cop_reward,
            thief_reward=thief_reward,
        )
