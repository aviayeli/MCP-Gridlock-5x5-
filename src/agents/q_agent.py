"""Tabular Q-learning agent: state, policy, Bellman update, JSON persistence.

State encoding, epsilon-greedy action selection, the TD-error update rule,
and save/load live together on `QLearningAgent`; the update-rule math and
the JSON (de)serialization format are each split into their own module
(`q_update.py`, `q_persistence.py`) purely to respect this package's
150-line-per-file cap -- see those modules for the WHY behind terminal-
state handling and the state-key JSON encoding, respectively.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from agents.q_persistence import load_q_table, save_q_table
from agents.q_update import apply_bellman_update
from agents.state_encoding import StateKey, encode_observation
from engine.player import Action, Role

if TYPE_CHECKING:
    from pathlib import Path

# Fixed action order used for deterministic argmax tie-breaking (first action
# in this order wins ties) and as the population for uniform random
# exploration draws. `tuple(Action)` preserves declaration order (UP, DOWN,
# LEFT, RIGHT, STAY) from `engine.player.Action`.
ACTIONS: tuple[Action, ...] = tuple(Action)


class QLearningAgent:
    """A single tabular Q-learning policy, shared by both Cop and Thief.

    One class parameterized by `Role` (not `CopAgent`/`ThiefAgent`
    subclasses) mirrors `engine.player.Player`'s own rationale: Cop and
    Thief differ only in which reward row/fog-of-war side applies -- both
    *caller* concerns, not behavioral differences in how a Q-learning
    policy looks up and selects actions.

    Hyperparameters (`alpha`, `gamma`, `epsilon`, ...) are constructor
    arguments, not values read from `config/game_config.json` -- that file
    describes the *game*, these describe the *learner* (same separation
    `config_loader.py` draws between JSON-driven `GameConfig` and the
    code-level `VISIBILITY_RADIUS` constant).
    """

    def __init__(
        self,
        role: Role,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_floor: float = 0.05,
        epsilon_decay: float = 0.9,
        rng: random.Random | None = None,
    ) -> None:
        """
        `gamma` defaults to 0.9 per the PRD's Dec-POMDP definition (§γ).
        `epsilon` starts high and is brought down by explicit
        `decay_epsilon()` calls (see that method). `rng` accepts an
        injected `random.Random`, the same determinism pattern
        `engine.board.Board` uses, so a caller can pin a seed and reproduce
        exploration draws (PRD §"Determinism & Reproducibility").
        """
        self.role = role
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_floor = epsilon_floor
        self.epsilon_decay = epsilon_decay
        self._rng = rng if rng is not None else random.Random()

        # Lazily populated: dict.fromkeys(ACTIONS, 0.0) per key means an
        # unseen state defaults to "all actions equally worthless" without
        # needing to pre-enumerate the ~224-key state space (see
        # state_encoding.py) up front.
        self.q_table: dict[StateKey, dict[Action, float]] = {}

    def encode_observation(self, observation: dict) -> StateKey:
        """Convert a wire-format observation dict into a Q-table state key.

        Thin delegation to `state_encoding.encode_observation` -- see that
        module for the full WHY behind the relative-position, fog-of-war
        -sentinel, and barrier-adjacency-mask design.
        """
        return encode_observation(observation)

    def q_values(self, state_key: StateKey) -> dict[Action, float]:
        """Return this state's per-action Q-values, creating them on first use."""
        return self.q_table.setdefault(state_key, dict.fromkeys(ACTIONS, 0.0))

    def epsilon_greedy_action(self, state_key: StateKey) -> Action:
        """Pick an action for `state_key` under the current epsilon.

        With probability `self.epsilon`, explore: return a uniformly random
        action from all 5 legal moves (including STAY). Otherwise, exploit:
        return the argmax-Q action, breaking ties deterministically by
        `ACTIONS` order (first action in fixed UP/DOWN/LEFT/RIGHT/STAY order
        among those tied for the max) so behavior is reproducible given the
        same `rng` draws and Q-table contents.
        """
        if self._rng.random() < self.epsilon:
            return self._rng.choice(ACTIONS)

        values = self.q_values(state_key)
        best_value = max(values.values())
        return next(action for action in ACTIONS if values[action] == best_value)

    def decay_epsilon(self) -> None:
        """Shrink epsilon toward `epsilon_floor` by one exponential-decay step.

        Caller-triggered only (never called implicitly from
        `epsilon_greedy_action`): "how many episodes have elapsed" is a
        tournament-level concept (PRD §"num_games") this agent shouldn't
        track itself, so the expected usage is one call per episode. Floored
        at `epsilon_floor` rather than zero so exploration never fully stops.
        """
        self.epsilon = max(self.epsilon_floor, self.epsilon * self.epsilon_decay)

    def update(
        self,
        state_key: StateKey,
        action: Action,
        reward: float,
        next_state_key: StateKey,
        done: bool,
    ) -> None:
        """`Q[s,a] += alpha * td_error`; see `q_update` for the terminal-state WHY."""
        apply_bellman_update(self, state_key, action, reward, next_state_key, done)

    def save(self, path: Path | str) -> None:
        """Persist `self.q_table` as JSON to `path`; see `q_persistence` for format.

        Caller convention (not enforced here): `data/q_table_team_alpha.json` /
        `data/q_table_team_beta.json`, one per team.
        """
        save_q_table(path, self.q_table)

    def load(self, path: Path | str) -> None:
        """Replace `self.q_table` from the JSON file at `path`.

        Missing file = cold start: `self.q_table` becomes `{}` instead of
        raising; see `q_persistence.load_q_table`.
        """
        self.q_table = load_q_table(path)
