"""The Bellman/TD-error tabular Q-learning update rule (PRD §{Ωi}).

Split out from q_agent.py purely to respect this package's 150-line-per-file
cap; `QLearningAgent.update` delegates here so the class body stays focused
on state/policy bookkeeping while the update-rule math (and its terminal-
state footgun) gets a module of its own.

WHY terminal transitions drop the discounted-future term entirely
--------------------------------------------------------------------
Standard one-step Q-learning is `Q[s,a] += alpha * td_error` with
`td_error = reward + gamma * max(Q[next_state]) - Q[state, action]`. That
formula assumes `next_state` is a real state the agent will actually
continue acting from. When `done` is True, the transition ended the episode
this turn (capture or move/turn-limit timeout, per `GameLoop`/`TurnResult`)
-- whatever `next_state_key` was computed from the terminal observation
does not correspond to a state any future action will be taken from, so
bootstrapping `gamma * max(Q[next_state_key])` into the target would leak
in Q-values from an unrelated future episode's starting position. The
standard (and easy to get backwards) fix is to drop the future term
entirely for terminal transitions: `td_error = reward - Q[state, action]`.
This function implements that by simply not calling `q_values` on
`next_state_key` at all when `done` is True, rather than calling it and
discarding the result -- so an unseen terminal `next_state_key` never even
gets a spurious all-zero entry lazily created in the Q-table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine.player import Action

if TYPE_CHECKING:
    from agents.q_agent import QLearningAgent
    from agents.state_encoding import StateKey


def apply_bellman_update(
    agent: QLearningAgent,
    state_key: StateKey,
    action: Action,
    reward: float,
    next_state_key: StateKey,
    done: bool,
) -> None:
    """Mutate `agent.q_table` in place per the one-step Q-learning update.

    Uses `agent.q_values(...)` for both `state_key` and (when not terminal)
    `next_state_key`, so a never-before-seen key lazily defaults to
    all-zero Q-values -- `max(...)` of an unseen next state is correctly 0.
    The dict returned by `q_values` is the same object stored in
    `agent.q_table`, so mutating it here mutates the table directly.
    """
    current_values = agent.q_values(state_key)
    future = 0.0 if done else agent.gamma * max(agent.q_values(next_state_key).values())
    td_error = reward + future - current_values[action]
    current_values[action] += agent.alpha * td_error
