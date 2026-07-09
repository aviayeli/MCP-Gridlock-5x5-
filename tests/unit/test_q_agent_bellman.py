"""Bellman update math tests for Q-learning agent (part 1: core hand-computed cases)."""

from __future__ import annotations

import pytest

from agents.q_agent import QLearningAgent
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
