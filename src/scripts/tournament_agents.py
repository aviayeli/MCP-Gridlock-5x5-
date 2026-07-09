"""Per-team Q-learning agent + policy-closure construction for the tournament runner.

Split out from `run_tournament.py` purely to respect this package's
150-line-per-file cap; "how does one team's Q-agent turn into a `Policy`
callable `Tournament` can call" is a self-contained concern, separate from
the top-level script orchestration (MCP smoke-check, `Tournament`
construction, report/Q-table persistence) `run_tournament.py` owns.

WHY one `QLearningAgent` per TEAM, not per role
-------------------------------------------------
`tournament_schedule.cop_team_for_game` swaps which *team* plays Cop every game —
Team Alpha is Cop on even-indexed games and Thief on odd-indexed games, and
vice versa for Team Beta (see `tournament.py`). A `Policy` is invoked with
whatever role's `Observation` its team currently holds, with no role tag in
sight. Since `agents.state_encoding.encode_observation` is already
role-agnostic by design (opponent encoded as a relative delta, not "the
opponent" vs. "the cop"/"the thief"), a per-team Q-table is exactly what's
being learned here: each team accumulates experience from *both* roles it
takes on across the series, onto one shared table, rather than splitting
that same experience across two tables that would each only ever see half
the games. `role=Role.COP` is passed to both teams' `QLearningAgent`
constructors below purely because the constructor requires *a* `Role` —
it is not semantically load-bearing for encoding or learning.

WHY the terminal-only, once-per-game update timing
-----------------------------------------------------
`TurnResult.cop_reward`/`.thief_reward` are `None` until the terminal turn
(this game's reward design is sparse/terminal-only — see `docs/PRD.md §R`);
there is no meaningful intermediate reward to bootstrap a per-turn update
from. `Tournament._play_game` also gives a `Policy` no way to learn *when*
a turn was terminal or what reward resulted — it is a stateless
`Observation -> Action` callable. So each `TeamAgent` remembers only its
own most recent `(state_key, action)` (updated every `policy()` call), and
performs exactly one Bellman update — using that remembered pair and the
team's actual terminal reward for the role it held that game — from
`Tournament`'s new `on_game_end` hook, once the game (and therefore the
reward) is known. `next_state_key` is irrelevant for a `done=True` update
(`q_update.apply_bellman_update` never looks at it in the terminal branch),
so the same `last_state` is reused rather than inventing a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.q_agent import QLearningAgent
from agents.state_encoding import StateKey
from engine.observation import Observation
from engine.player import Action, Role
from engine.tournament_report import GameRecord, Team


def _to_wire_format(obs: Observation) -> dict:
    """Bridge the engine's internal `Observation` dataclass to the MCP
    wire-format dict `QLearningAgent.encode_observation` expects.

    `Tournament` hands a `Policy` the engine-internal `Observation`
    dataclass (tuple positions, a `frozenset` of barriers) — not the JSON
    wire-format dict (`list` positions, `list`-of-`list` barriers) that
    `GameTools.get_observation` serializes for real MCP clients. Rather
    than import `mcp_server.tools._serialize_observation` (private, and a
    layering inversion — this script driving `Tournament` directly has no
    business depending on the MCP transport package), this is a small,
    local, pure equivalent.
    """
    opponent = obs.opponent_position
    return {
        "self_position": list(obs.self_position),
        "opponent_position": list(opponent) if opponent is not None else None,
        "barriers": [list(cell) for cell in obs.barriers],
        "move_count": obs.move_count,
    }


@dataclass
class TeamAgent:
    """One team's `QLearningAgent` plus its most-recent `(state, action)`.

    The `last_state`/`last_action` fields are simple mutable closure state
    (per the module docstring's WHY) rather than anything `Tournament`
    tracks on the team's behalf — `Tournament` has no concept of "policy
    state" beyond the `Policy` callable itself.
    """

    team: Team
    agent: QLearningAgent
    last_state: StateKey | None = None
    last_action: Action | None = None

    def policy(self, obs: Observation) -> Action:
        """The actual `Policy` callable: encode, act epsilon-greedily, remember."""
        state_key = self.agent.encode_observation(_to_wire_format(obs))
        action = self.agent.epsilon_greedy_action(state_key)
        self.last_state, self.last_action = state_key, action
        return action

    def learn_from(self, record: GameRecord) -> None:
        """Perform this team's one terminal Q-update for `record`, then decay epsilon.

        Called from `Tournament`'s `on_game_end` hook. Reward is the team's
        actual terminal payout for the role it held this game (Cop's
        `cop_reward` if this team was cop_team, else Thief's
        `thief_reward`) — see module docstring for why this is the only
        well-defined update per game.
        """
        if self.last_state is not None and self.last_action is not None:
            reward = (
                record.cop_reward
                if self.team is record.cop_team
                else record.thief_reward
            )
            self.agent.update(
                self.last_state, self.last_action, reward, self.last_state, done=True
            )
        self.agent.decay_epsilon()


def build_team_agents() -> dict[Team, TeamAgent]:
    """Construct one fresh `TeamAgent` per `Team`, ready to hand to `Tournament`.

    `role=Role.COP` is passed to both constructors — see module docstring's
    first WHY for why this is not semantically meaningful here.
    """
    return {
        team: TeamAgent(team=team, agent=QLearningAgent(role=Role.COP))
        for team in Team
    }
