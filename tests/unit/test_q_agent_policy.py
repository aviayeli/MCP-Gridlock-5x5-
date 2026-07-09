"""Epsilon-greedy policy tests for Q-learning agent."""

from __future__ import annotations

import random

from agents.q_agent import ACTIONS, QLearningAgent
from agents.state_encoding import StateKey
from engine.player import Action, Role


class TestEpsilonGreedyPolicy:
    """Epsilon-greedy action selection and tie-breaking."""

    def test_epsilon_zero_always_exploit(self) -> None:
        """epsilon=0.0: always picks the argmax-Q action."""
        agent = QLearningAgent(Role.COP, epsilon=0.0)
        state_key: StateKey = ((1, 0), 0)
        # Manually set Q-values to create a clear max
        agent.q_table[state_key] = {
            Action.UP: 10.0,
            Action.DOWN: 5.0,
            Action.LEFT: 3.0,
            Action.RIGHT: 8.0,
            Action.STAY: 2.0,
        }
        # With epsilon=0, should always pick UP (the max)
        for _ in range(10):
            assert agent.epsilon_greedy_action(state_key) == Action.UP

    def test_epsilon_zero_tie_break_first_in_order(self) -> None:
        """epsilon=0.0: tie-breaking picks first action in ACTIONS order."""
        agent = QLearningAgent(Role.COP, epsilon=0.0)
        state_key: StateKey = ((1, 0), 0)
        # All actions tied at same Q-value
        agent.q_table[state_key] = {
            Action.UP: 5.0,
            Action.DOWN: 5.0,
            Action.LEFT: 5.0,
            Action.RIGHT: 5.0,
            Action.STAY: 5.0,
        }
        # Should always pick first in ACTIONS order (UP)
        assert agent.epsilon_greedy_action(state_key) == Action.UP

    def test_epsilon_zero_partial_tie_break(self) -> None:
        """epsilon=0.0: among tied-max actions, first in ACTIONS order wins."""
        agent = QLearningAgent(Role.COP, epsilon=0.0)
        state_key: StateKey = ((1, 0), 0)
        # DOWN and STAY tied for max
        agent.q_table[state_key] = {
            Action.UP: 3.0,
            Action.DOWN: 10.0,
            Action.LEFT: 5.0,
            Action.RIGHT: 2.0,
            Action.STAY: 10.0,
        }
        # STAY is tied with DOWN but comes later in ACTIONS order,
        # so DOWN (first in order among max values) should win
        assert agent.epsilon_greedy_action(state_key) == Action.DOWN

    def test_epsilon_one_always_explore(self) -> None:
        """epsilon=1.0: always picks a uniformly random action."""
        rng = random.Random(42)
        agent = QLearningAgent(Role.COP, epsilon=1.0, rng=rng)
        state_key: StateKey = ((1, 0), 0)
        # With epsilon=1, result should be a valid Action
        agent.q_table[state_key] = {
            Action.UP: 100.0,
            Action.DOWN: 1.0,
            Action.LEFT: 1.0,
            Action.RIGHT: 1.0,
            Action.STAY: 1.0,
        }
        actions = set()
        for _ in range(100):
            action = agent.epsilon_greedy_action(state_key)
            assert action in ACTIONS
            actions.add(action)
        # With 100 draws and epsilon=1, likely to see multiple different actions
        assert len(actions) > 1

    def test_unseen_state_defaults_to_zero_q_values(self) -> None:
        """Fresh/never-seen state defaults to all-zero Q-values."""
        agent = QLearningAgent(Role.COP, epsilon=0.0)
        state_key: StateKey = ((2, -1), 5)
        # Never explicitly set q_table[state_key]
        values = agent.q_values(state_key)
        assert all(v == 0.0 for v in values.values())

    def test_unseen_state_with_epsilon_zero_returns_first_action(self) -> None:
        """Unseen state with epsilon=0 returns first-order action."""
        agent = QLearningAgent(Role.COP, epsilon=0.0)
        state_key: StateKey = ((1, 1), 3)
        # All zero Q-values; argmax tie means first in ACTIONS order
        action = agent.epsilon_greedy_action(state_key)
        assert action == Action.UP  # First in ACTIONS order

    def test_decay_epsilon_exponential_decay(self) -> None:
        """decay_epsilon() reduces epsilon, never below floor."""
        agent = QLearningAgent(
            Role.COP,
            epsilon=1.0,
            epsilon_floor=0.05,
            epsilon_decay=0.9,
        )
        original = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon == original * 0.9
        assert 0.05 <= agent.epsilon <= 1.0

    def test_decay_epsilon_never_below_floor(self) -> None:
        """decay_epsilon(): epsilon stays >= epsilon_floor even after many decays."""
        agent = QLearningAgent(
            Role.COP,
            epsilon=0.1,
            epsilon_floor=0.05,
            epsilon_decay=0.5,
        )
        for _ in range(10):
            agent.decay_epsilon()
        assert agent.epsilon >= agent.epsilon_floor

    def test_decay_epsilon_exact_formula_one_step(self) -> None:
        """decay_epsilon(): hand-compute one step to verify exact formula."""
        agent = QLearningAgent(
            Role.COP,
            epsilon=1.0,
            epsilon_floor=0.05,
            epsilon_decay=0.9,
        )
        expected = max(0.05, 1.0 * 0.9)
        agent.decay_epsilon()
        assert agent.epsilon == expected

    def test_epsilon_greedy_with_seeded_rng(self) -> None:
        """epsilon=1.0 with seeded rng produces deterministic action sequences."""
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        agent1 = QLearningAgent(Role.COP, epsilon=1.0, rng=rng1)
        agent2 = QLearningAgent(Role.COP, epsilon=1.0, rng=rng2)
        state_key: StateKey = ((1, 0), 0)

        actions1 = [agent1.epsilon_greedy_action(state_key) for _ in range(20)]
        actions2 = [agent2.epsilon_greedy_action(state_key) for _ in range(20)]
        assert actions1 == actions2
