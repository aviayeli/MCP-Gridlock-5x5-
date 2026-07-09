"""Bellman update and Q-value learning tests for Q-learning agent."""

from __future__ import annotations

import pytest

from agents.q_agent import QLearningAgent
from agents.q_update import apply_bellman_update
from agents.state_encoding import StateKey
from engine.player import Action, Role


class TestBellmanUpdateMath:
    """Bellman/TD-error Q-value updates, the most critical correctness test."""

    def test_update_non_terminal_unseen_next_state(self) -> None:
        """Non-terminal unseen next state: td_error uses max(Q[next]) = 0."""
        agent = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        state_key: StateKey = ((1, 0), 0)
        next_state_key: StateKey = ((0, 1), 0)
        action = Action.UP
        reward = 5.0
        # Initial Q[s,a] is 0.0
        initial_q = agent.q_values(state_key)[action]
        assert initial_q == 0.0
        # Apply update: td_error = reward + gamma * max(Q[next]) - Q[s,a]
        #                        = 5.0 + 0.9 * 0.0 - 0.0 = 5.0
        # Q[s,a] += 0.1 * 5.0 = 0.5
        agent.update(state_key, action, reward, next_state_key, done=False)
        assert agent.q_table[state_key][action] == pytest.approx(0.5)
    def test_update_non_terminal_with_seeded_next_state(self) -> None:
        """Non-terminal update bootstraps correctly: includes gamma * max(Q[next])."""
        agent = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        state_key: StateKey = ((1, 0), 0)
        next_state_key: StateKey = ((0, 1), 0)
        action = Action.UP
        reward = 5.0
        # Pre-seed next_state with a known max Q-value
        agent.q_table[next_state_key] = {
            Action.UP: 10.0,
            Action.DOWN: 5.0,
            Action.LEFT: 3.0,
            Action.RIGHT: 7.0,
            Action.STAY: 2.0,
        }
        # td_error = 5.0 + 0.9 * 10.0 - 0.0 = 5.0 + 9.0 = 14.0
        # Q[s,a] += 0.1 * 14.0 = 1.4
        agent.update(state_key, action, reward, next_state_key, done=False)
        assert agent.q_table[state_key][action] == pytest.approx(1.4)
    def test_update_terminal_no_future_term(self) -> None:
        """Terminal update excludes future term: td_error = reward - Q[s,a]."""
        agent = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        state_key: StateKey = ((1, 0), 0)
        next_state_key: StateKey = ((0, 1), 0)
        action = Action.UP
        reward = 5.0
        # Pre-seed next_state with a high Q-value to verify it's *not* bootstrapped
        agent.q_table[next_state_key] = {
            Action.UP: 100.0,
            Action.DOWN: 100.0,
            Action.LEFT: 100.0,
            Action.RIGHT: 100.0,
            Action.STAY: 100.0,
        }
        # td_error = 5.0 - 0.0 = 5.0 (no gamma * max(Q[next]) term)
        # Q[s,a] += 0.1 * 5.0 = 0.5
        agent.update(state_key, action, reward, next_state_key, done=True)
        assert agent.q_table[state_key][action] == pytest.approx(0.5)
    def test_terminal_vs_non_terminal_differ(self) -> None:
        """Terminal and non-terminal updates differ (critical bug catch)."""
        # Non-terminal update
        agent_nonterminal = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        state_key: StateKey = ((1, 0), 0)
        next_state_key: StateKey = ((0, 1), 0)
        action = Action.UP
        reward = 5.0
        agent_nonterminal.q_table[next_state_key] = {
            Action.UP: 10.0,
            Action.DOWN: 5.0,
            Action.LEFT: 3.0,
            Action.RIGHT: 7.0,
            Action.STAY: 2.0,
        }
        agent_nonterminal.update(state_key, action, reward, next_state_key, done=False)
        nonterminal_value = agent_nonterminal.q_table[state_key][action]
        # Terminal update with identical setup except done=True
        agent_terminal = QLearningAgent(Role.COP, alpha=0.1, gamma=0.9)
        agent_terminal.q_table[next_state_key] = {
            Action.UP: 10.0,
            Action.DOWN: 5.0,
            Action.LEFT: 3.0,
            Action.RIGHT: 7.0,
            Action.STAY: 2.0,
        }
        agent_terminal.update(state_key, action, reward, next_state_key, done=True)
        terminal_value = agent_terminal.q_table[state_key][action]
        # They must differ (terminal should be 0.5, nonterminal should be 1.4)
        assert nonterminal_value != terminal_value
        assert terminal_value < nonterminal_value
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
