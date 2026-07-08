"""Base Q-learning agent: state encoding + epsilon-greedy policy (PRD §{Ωi}).

This is Step 1 of the Q-learning agent build-out: state encoding and action
selection only. The Bellman/TD-error update rule and Q-table persistence
(save/load) are an explicitly separate, later step -- see the class
docstring below -- so `q_table` is populated lazily here but never mutated
by anything in this file beyond that lazy default-initialization.
"""

from __future__ import annotations

import random

from agents.state_encoding import StateKey, encode_observation
from engine.player import Action, Role

# Fixed action order used for deterministic argmax tie-breaking (first action
# in this order wins ties) and as the population for uniform random
# exploration draws. `tuple(Action)` preserves declaration order (UP, DOWN,
# LEFT, RIGHT, STAY) from `engine.player.Action`.
ACTIONS: tuple[Action, ...] = tuple(Action)


class QLearningAgent:
    """A single tabular Q-learning policy, shared by both Cop and Thief.

    One class parameterized by `Role` (not `CopAgent`/`ThiefAgent`
    subclasses) mirrors `engine.player.Player`'s own documented rationale:
    Cop and Thief differ only in which reward row applies and which side of
    the fog-of-war they sit on -- both are concerns of the *caller* (the
    tournament/game_loop hands each role its own observation and reward),
    not behavioral differences in how a Q-learning policy looks up and
    selects actions. A subclass split would duplicate identical
    state-encoding and epsilon-greedy logic for no gain.

    Hyperparameters (`alpha`, `gamma`, `epsilon`, ...) are constructor
    arguments with sensible defaults, not values read from
    `config/game_config.json` -- that file describes the *game* (grid size,
    scoring, move/game limits), while these describe the *learner*, exactly
    the same separation `config_loader.py` already draws between JSON-driven
    `GameConfig` and the module-level `VISIBILITY_RADIUS` constant that
    lives in code instead.
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
        `gamma` defaults to 0.9 to match the PRD's Dec-POMDP definition
        (§γ). `epsilon` starts high (full exploration) and is brought down
        toward `epsilon_floor` by explicit calls to `decay_epsilon()` --
        see that method's docstring for why decay is caller-triggered
        rather than automatic. `rng` accepts an injected `random.Random`
        (rather than reseeding the global `random` module), the same
        determinism pattern `engine.board.Board` uses for barrier
        placement, so a caller can pin a seed and reproduce identical
        exploration draws across a run (PRD §"Determinism &
        Reproducibility").
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
        # state_encoding.py) up front. Step 2 (Bellman update) will mutate
        # the inner per-action values this class only reads/defaults here.
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

        Contract: this method mutates `epsilon` only when explicitly called
        by the caller -- it is never invoked implicitly from
        `epsilon_greedy_action`. Per-episode (rather than per-action) decay
        is the caller's responsibility because "how many episodes have
        elapsed" is a tournament-level concept (PRD §"num_games") that this
        agent has no visibility into and should not track itself; the
        expected usage is one `decay_epsilon()` call between episodes, e.g.
        once per game in a `num_games`-game series, so exploration cools
        down gradually across the run rather than within a single episode's
        handful of moves. The schedule is simple exponential decay,
        `epsilon *= epsilon_decay`, clamped at `epsilon_floor` so the agent
        never stops exploring entirely (a floor above zero keeps a
        already-converged Q-table from becoming permanently stuck if the
        environment's dynamics were ever to shift).
        """
        self.epsilon = max(self.epsilon_floor, self.epsilon * self.epsilon_decay)
