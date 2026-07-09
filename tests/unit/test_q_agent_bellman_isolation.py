"""Bellman update math tests for Q-learning agent (part 2: compounding/isolation)."""

from __future__ import annotations

import pytest

from agents.q_agent import QLearningAgent
from agents.q_update import apply_bellman_update
from agents.state_encoding import StateKey
from engine.player import Action, Role


class TestBellmanUpdateIsolation:
    """Sequential compounding, cross-action/state isolation, and edge-case rewards."""

    def test_multiple_sequential_updates_compound(self) -> None:
        """Multiple sequential updates to same (state, action) compound correctly."""
        agent = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        state_key: StateKey = ((1, 0), 0)
        next_state_key: StateKey = ((0, 1), 0)
        action = Action.UP
        # First update: Q[s,a] = 0.0 + 0.1 * (5.0 - 0.0) = 0.5
        agent.update(state_key, action, 5.0, next_state_key, done=False)
        q_after_1 = agent.q_table[state_key][action]
        assert q_after_1 == pytest.approx(0.5)
        # Second update: Q[s,a] = 0.5 + 0.1 * (3.0 - 0.5) = 0.5 + 0.25 = 0.75
        agent.update(state_key, action, 3.0, next_state_key, done=False)
        q_after_2 = agent.q_table[state_key][action]
        assert q_after_2 == pytest.approx(0.75)
        # Third update: Q[s,a] = 0.75 + 0.1 * (2.0 - 0.75) = 0.75 + 0.125 = 0.875
        agent.update(state_key, action, 2.0, next_state_key, done=False)
        q_after_3 = agent.q_table[state_key][action]
        assert q_after_3 == pytest.approx(0.875)

    def test_update_one_action_does_not_affect_others(self) -> None:
        """Updating one action does not affect other actions in the same state."""
        agent = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        state_key: StateKey = ((1, 0), 0)
        next_state_key: StateKey = ((0, 1), 0)
        # Update only UP
        agent.update(state_key, Action.UP, 5.0, next_state_key, done=False)
        # DOWN, LEFT, RIGHT, STAY should still be 0.0
        assert agent.q_table[state_key][Action.DOWN] == 0.0
        assert agent.q_table[state_key][Action.LEFT] == 0.0
        assert agent.q_table[state_key][Action.RIGHT] == 0.0
        assert agent.q_table[state_key][Action.STAY] == 0.0

    def test_update_one_state_does_not_affect_others(self) -> None:
        """Updating one state does not affect other states' values."""
        agent = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        state1: StateKey = ((1, 0), 0)
        state2: StateKey = ((0, 1), 0)
        next_state: StateKey = ((2, 2), 1)
        # Update state1
        agent.update(state1, Action.UP, 5.0, next_state, done=False)
        # state2 should not exist in q_table yet
        assert state2 not in agent.q_table
        # Manually check it defaults to zero if accessed
        assert agent.q_values(state2)[Action.UP] == 0.0

    def test_apply_bellman_update_directly(self) -> None:
        """Test apply_bellman_update function directly."""
        agent = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        state_key: StateKey = ((1, 0), 0)
        next_state_key: StateKey = ((0, 1), 0)
        action = Action.UP
        reward = 5.0
        apply_bellman_update(agent, state_key, action, reward, next_state_key, False)
        assert agent.q_table[state_key][action] == pytest.approx(0.5)

    def test_update_with_negative_reward(self) -> None:
        """Updates with negative rewards work correctly (penalizing actions)."""
        agent = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        state_key: StateKey = ((1, 0), 0)
        next_state_key: StateKey = ((0, 1), 0)
        action = Action.UP
        # td_error = -10.0 + 0.9 * 0.0 - 0.0 = -10.0
        # Q[s,a] += 0.1 * (-10.0) = -1.0
        agent.update(state_key, action, -10.0, next_state_key, done=False)
        assert agent.q_table[state_key][action] == pytest.approx(-1.0)

    def test_update_with_zero_reward(self) -> None:
        """Updates with zero reward work (typically for non-terminal steps)."""
        agent = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        state_key: StateKey = ((1, 0), 0)
        next_state_key: StateKey = ((0, 1), 0)
        action = Action.UP
        # With 0 reward and unseen next state: td_error = 0.0 + 0.0 - 0.0 = 0.0
        agent.update(state_key, action, 0.0, next_state_key, done=False)
        assert agent.q_table[state_key][action] == 0.0
