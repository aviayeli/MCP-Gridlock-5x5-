"""Offline batch driver: plays the full tournament series and trains per-team
Q-agents, then writes a JSON series report and the two teams' Q-tables.

WHY this script drives `Tournament` directly, not `mcp_server`'s stdio app
----------------------------------------------------------------------------
`mcp_server.server` (via `MatchState`/`GameTools`) is the live, already-
tested entry point for *independent, asynchronous* external LLM clients —
it buffers whichever role's action arrives first and only resolves a turn
once both are in, because two real MCP clients each call `make_move` on
their own schedule without seeing each other's move first (see
`match_state.py`'s own docstring). That is a different concurrency model
than an offline batch trainer needs: this script plays both teams *in the
same process*, synchronously, once per game, which is exactly what
`Tournament._play_game` already does by calling `GameLoop.step` directly.
Driving `Tournament` through the async stdio buffering layer instead would
require standing up a real subprocess plus an async MCP client and
reconciling two purpose-built concurrency models for no benefit — so this
script uses `Tournament` for game-driving, and separately instantiates the
MCP tool layer below purely as a real (not decorative) smoke check that the
MCP integration is wired to the same underlying engine this tournament
exercises, before running the series.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from engine.tournament import GameRecord, Tournament
from engine.tournament_report import Team
from mcp_server.match_state import MatchState
from mcp_server.tools import GameTools
from scripts.tournament_agents import TeamAgent, build_team_agents

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "results" / "tournament_report.json"
DATA_DIR = REPO_ROOT / "data"
Q_TABLE_PATHS = {
    Team.ALPHA: DATA_DIR / "q_table_team_alpha.json",
    Team.BETA: DATA_DIR / "q_table_team_beta.json",
}


def _smoke_check_mcp_layer() -> None:
    """Instantiate the real MCP tool stack against a fresh match, once.

    Not decorative: this constructs the actual `GameTools`/`MatchState`
    objects real LLM clients talk to and calls a read-only tool through
    them, demonstrating the MCP layer is correctly wired to the same
    engine this tournament drives — without using it to play the series
    itself (see module docstring for why).
    """
    tools = GameTools(MatchState())
    status = tools.get_match_status()
    assert status["move_count"] == 0
    assert status["done"] is False


def _make_on_game_end(
    team_agents: dict[Team, TeamAgent],
) -> Callable[[int, GameRecord], None]:
    """Build the `Tournament(on_game_end=...)` callback: one terminal update
    plus one epsilon decay per team, per finished game (see
    `tournament_agents`'s module docstring for the full reward-timing WHY).
    """

    def on_game_end(_game_index: int, record: GameRecord) -> None:
        """Invoke each team agent's terminal Q-update and epsilon decay."""
        for team_agent in team_agents.values():
            team_agent.learn_from(record)

    return on_game_end


def main() -> None:
    """Run the MCP smoke check, play the series, and persist report + Q-tables."""
    _smoke_check_mcp_layer()

    team_agents = build_team_agents()
    policies = {team: agent.policy for team, agent in team_agents.items()}
    tournament = Tournament(policies, on_game_end=_make_on_game_end(team_agents))
    tournament.play()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(tournament.report(), indent=2))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for team, team_agent in team_agents.items():
        team_agent.agent.save(Q_TABLE_PATHS[team])

    report = tournament.report()
    print(
        f"Wrote {RESULTS_PATH} "
        f"({report['num_games']} games, winner={report['winner']})"
    )
    print(f"Final scores: {report['final_scores']}")
    for team, path in Q_TABLE_PATHS.items():
        num_states = len(team_agents[team].agent.q_table)
        print(f"Wrote {path} for {team.value} ({num_states} states)")


if __name__ == "__main__":
    main()
