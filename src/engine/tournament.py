"""Multi-game tournament driver: plays a `config.num_games`-game series
between two policy-driven teams, alternating which team plays Cop vs Thief
each game and tracking cumulative per-team score.

This module deliberately knows nothing about *how* a policy decides a move
(Q-table, LLM call, fixed heuristic) — see the `Policy` contract below —
so a future phase can wire in real LLM-driven agents without touching
anything here. Report assembly is split out to tournament_report.py purely
to respect this package's 150-line-per-file cap.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from engine.board import Coordinate
from engine.config_loader import GameConfig, load_config
from engine.game_loop import GameLoop
from engine.observation import Observation, build_observation
from engine.player import Action
from engine.tournament_report import GameRecord, Team, build_game_record, build_report

Policy = Callable[[Observation], Action]
"""Interface contract for a pluggable per-team strategy.

A policy is any callable that takes the single `Observation` its role
currently sees (own position, opponent position or `UNSEEN` if out of the
fog-of-war radius, the static barrier layout, and the current move count)
and returns the `Action` to submit for this turn. Policies are role- and
team-agnostic at the type level — the same callable shape plays Cop in one
game and Thief in the next under the role-swap schedule below, exactly
because `Observation` is already relative to whichever role holds it. This
is the *only* seam `Tournament` exposes to its caller: a test can inject a
trivial deterministic policy (e.g. always `Action.STAY`, or "step toward
the last-seen opponent"), and a later phase can inject a real LLM- or
Q-table-driven policy, without either side changing anything in this file.
"""


def _cop_team_for_game(game_index: int) -> Team:
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
    ) -> None:
        """`policies` must supply exactly one entry per `Team` member.

        A dict keyed by `Team` (rather than two positional arguments) is
        used so the mapping stays legible at call sites and so a future
        team roster change (should one ever happen) would not shift
        argument order silently.
        """
        if set(policies) != set(Team):
            raise ValueError("policies must provide exactly one entry per Team")
        self.config = config or load_config()
        self.policies = policies
        self.cop_start = cop_start
        self.thief_start = thief_start
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
            self.records.append(self._play_game(game_index))

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
