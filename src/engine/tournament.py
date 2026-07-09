"""Multi-game tournament driver: plays a `config.num_games`-game series
between two policy-driven teams, alternating which team plays Cop vs Thief
each game and tracking cumulative per-team score.

This module deliberately knows nothing about *how* a policy decides a move
(Q-table, LLM call, fixed heuristic) — see the `Policy` contract in
tournament_policy.py — so a future phase can wire in real LLM-driven agents
without touching anything here. Report assembly and the role-swap schedule
are split out to tournament_report.py and tournament_schedule.py purely to
respect this package's 150-line-per-file cap.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from engine.board import Coordinate
from engine.config_loader import GameConfig, load_config
from engine.game_loop import GameLoop
from engine.observation import build_observation
from engine.tournament_policy import Policy
from engine.tournament_report import GameRecord, Team, build_game_record, build_report
from engine.tournament_schedule import _cop_team_for_game


class Tournament:
    """Plays and scores a `config.num_games`-game series between two teams.

    Owns the outer game-index loop and the role-swap schedule; delegates
    single-episode play to a fresh `GameLoop` per game (mirroring how
    `GameLoop` itself delegates turn resolution to `Board`/`Player`) and
    delegates report assembly to `tournament_report.build_report`.
    """

    def __init__(
        self,
        policies: dict[Team, Policy],
        config: GameConfig | None = None,
        cop_start: Coordinate = (0, 0),
        thief_start: Coordinate = (4, 4),
        on_game_end: Callable[[int, GameRecord], None] | None = None,
    ) -> None:
        """`policies` must supply exactly one entry per `Team` member.

        A dict keyed by `Team` (rather than two positional arguments) is
        used so the mapping stays legible at call sites and so a future
        team roster change (should one ever happen) would not shift
        argument order silently.

        `on_game_end`, if given, is called once per finished game as
        `on_game_end(game_index, record)`, right after that game's record
        is appended in `play()`. This is the one hook `Policy` itself
        cannot provide (a stateless `Observation -> Action` callable has no
        channel back to "here's the reward" or "the episode just ended") —
        it lets a caller do per-episode bookkeeping (e.g. decaying an
        epsilon-greedy agent, or a sparse terminal-reward Q-update) without
        `Tournament` knowing anything about learning. Defaults to `None` so
        existing callers are unaffected.
        """
        if set(policies) != set(Team):
            raise ValueError("policies must provide exactly one entry per Team")
        self.config = config or load_config()
        self.policies = policies
        self.cop_start = cop_start
        self.thief_start = thief_start
        self.on_game_end = on_game_end
        self.team_scores: dict[Team, int] = dict.fromkeys(Team, 0)
        self.records: list[GameRecord] = []

    def play(self) -> None:
        """Play all `config.num_games` games in order, in a single pass.

        Calling `play()` more than once on the same instance would append
        duplicate game_index records and double-count cumulative scores,
        so — like `GameLoop.step` refusing to run past `done` — callers
        are expected to call this exactly once per `Tournament` instance
        rather than have it silently reset state on a second call.
        """
        for game_index in range(self.config.num_games):
            record = self._play_game(game_index)
            self.records.append(record)
            if self.on_game_end is not None:
                self.on_game_end(game_index, record)

    def _play_game(self, game_index: int) -> GameRecord:
        """Drive one full episode to completion via the injected policies."""
        cop_team = _cop_team_for_game(game_index)
        thief_team = Team.BETA if cop_team is Team.ALPHA else Team.ALPHA
        loop = GameLoop(self.config, self.cop_start, self.thief_start)

        barriers = loop.board.barrier_coordinates()
        cop_obs = build_observation(
            loop.cop.position, loop.thief.position, barriers, loop.move_count
        )
        thief_obs = build_observation(
            loop.thief.position, loop.cop.position, barriers, loop.move_count
        )

        result = None
        while result is None or not result.done:
            cop_action = self.policies[cop_team](cop_obs)
            thief_action = self.policies[thief_team](thief_obs)
            result = loop.step(cop_action, thief_action)
            cop_obs, thief_obs = result.cop_observation, result.thief_observation

        record = build_game_record(
            game_index, cop_team, thief_team, result, self.config
        )
        self.team_scores[cop_team] += record.cop_reward
        self.team_scores[thief_team] += record.thief_reward
        return record

    def report(self) -> dict:
        """Return the series report; see `tournament_report.build_report`."""
        return build_report(self.config, self.records, self.team_scores)

    def to_json(self) -> str:
        """Convenience wrapper: `report()` rendered as indented JSON."""
        return json.dumps(self.report(), indent=2)
