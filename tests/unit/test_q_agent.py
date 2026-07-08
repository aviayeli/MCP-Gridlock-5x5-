"""Comprehensive tests for Q-learning agent.

Tests cover state encoding, epsilon-greedy policy, Bellman updates, and persistence.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from agents.q_agent import ACTIONS, QLearningAgent
from agents.q_persistence import (
    _decode_state_key,
    _decode_values,
    _encode_state_key,
    _encode_values,
    load_q_table,
    save_q_table,
)
from agents.q_update import apply_bellman_update
from agents.state_encoding import StateKey, encode_observation
from engine.player import Action, Role


class TestStateEncodingDeterminism:
    """State serialization handles different fog-of-war views deterministically."""

    def test_identical_observations_encode_identically(self) -> None:
        """Identical observation dicts always encode to the identical StateKey."""
        obs1 = {
            "self_position": [0, 0],
            "opponent_position": [1, 0],
            "barriers": [],
            "move_count": 5,
        }
        obs2 = {
            "self_position": [0, 0],
            "opponent_position": [1, 0],
            "barriers": [],
            "move_count": 5,
        }
        assert encode_observation(obs1) == encode_observation(obs2)

    def test_barrier_order_independence(self) -> None:
        """Barriers in different list order encode to the same key."""
        obs1 = {
            "self_position": [0, 0],
            "opponent_position": [1, 0],
            "barriers": [[0, 1], [1, 1]],
            "move_count": 5,
        }
        obs2 = {
            "self_position": [0, 0],
            "opponent_position": [1, 0],
            "barriers": [[1, 1], [0, 1]],  # Different order
            "move_count": 5,
        }
        assert encode_observation(obs1) == encode_observation(obs2)

    def test_visible_vs_unseen_opponent_different_keys(self) -> None:
        """Visible vs unseen opponent produce different, distinguishable keys."""
        obs_visible = {
            "self_position": [0, 0],
            "opponent_position": [1, 0],
            "barriers": [],
            "move_count": 5,
        }
        obs_unseen = {
            "self_position": [0, 0],
            "opponent_position": None,
            "barriers": [],
            "move_count": 5,
        }
        key_visible = encode_observation(obs_visible)
        key_unseen = encode_observation(obs_unseen)
        assert key_visible != key_unseen
        # Unseen key's first element must be exactly None
        assert key_unseen[0] is None

    def test_unseen_key_first_element_is_none(self) -> None:
        """Unseen opponent key's first element is exactly None (not a tuple)."""
        obs = {
            "self_position": [2, 2],
            "opponent_position": None,
            "barriers": [],
            "move_count": 0,
        }
        key = encode_observation(obs)
        assert key[0] is None
        assert isinstance(key[1], int)  # Second element is barrier mask

    def test_same_relative_delta_at_different_positions_same_key(self) -> None:
        """Same relative delta at different positions encode identically."""
        # Both have opponent one cell down from self
        obs1 = {
            "self_position": [0, 0],
            "opponent_position": [1, 0],
            "barriers": [],
            "move_count": 0,
        }
        obs2 = {
            "self_position": [3, 3],
            "opponent_position": [4, 3],
            "barriers": [],
            "move_count": 0,
        }
        assert encode_observation(obs1) == encode_observation(obs2)

    def test_different_relative_deltas_different_keys(self) -> None:
        """Different relative opponent deltas produce different keys."""
        obs_down = {
            "self_position": [0, 0],
            "opponent_position": [1, 0],
            "barriers": [],
            "move_count": 0,
        }
        obs_right = {
            "self_position": [0, 0],
            "opponent_position": [0, 1],
            "barriers": [],
            "move_count": 0,
        }
        assert encode_observation(obs_down) != encode_observation(obs_right)

    def test_barrier_adjacency_up_vs_down(self) -> None:
        """Barrier immediately UP vs. immediately DOWN produce different mask values."""
        obs_up = {
            "self_position": [1, 1],
            "opponent_position": None,
            "barriers": [[0, 1]],  # One cell UP
            "move_count": 0,
        }
        obs_down = {
            "self_position": [1, 1],
            "opponent_position": None,
            "barriers": [[2, 1]],  # One cell DOWN
            "move_count": 0,
        }
        key_up = encode_observation(obs_up)
        key_down = encode_observation(obs_down)
        assert key_up != key_down
        assert key_up[1] != key_down[1]  # Different barrier masks

    def test_barrier_adjacency_left_vs_right(self) -> None:
        """LEFT vs RIGHT barrier produce different mask values."""
        obs_left = {
            "self_position": [1, 1],
            "opponent_position": None,
            "barriers": [[1, 0]],  # One cell LEFT
            "move_count": 0,
        }
        obs_right = {
            "self_position": [1, 1],
            "opponent_position": None,
            "barriers": [[1, 2]],  # One cell RIGHT
            "move_count": 0,
        }
        key_left = encode_observation(obs_left)
        key_right = encode_observation(obs_right)
        assert key_left != key_right
        assert key_left[1] != key_right[1]  # Different barrier masks

    def test_non_adjacent_barrier_ignored(self) -> None:
        """Barrier 2+ cells away does not affect the mask (locality test)."""
        obs_no_barrier = {
            "self_position": [1, 1],
            "opponent_position": None,
            "barriers": [],
            "move_count": 0,
        }
        obs_far_barrier = {
            "self_position": [1, 1],
            "opponent_position": None,
            "barriers": [[0, 0]],  # Two cells away (diagonal)
            "move_count": 0,
        }
        assert encode_observation(obs_no_barrier) == encode_observation(obs_far_barrier)

    def test_move_count_excluded_from_key(self) -> None:
        """Different move_count values do not change the state key."""
        obs1 = {
            "self_position": [0, 0],
            "opponent_position": [1, 0],
            "barriers": [],
            "move_count": 5,
        }
        obs2 = {
            "self_position": [0, 0],
            "opponent_position": [1, 0],
            "barriers": [],
            "move_count": 10,
        }
        assert encode_observation(obs1) == encode_observation(obs2)

    def test_multiple_adjacent_barriers_mask(self) -> None:
        """Multiple adjacent barriers compose correctly into mask bits."""
        obs_up_and_left = {
            "self_position": [1, 1],
            "opponent_position": None,
            "barriers": [[0, 1], [1, 0]],  # UP and LEFT
            "move_count": 0,
        }
        obs_down_and_right = {
            "self_position": [1, 1],
            "opponent_position": None,
            "barriers": [[2, 1], [1, 2]],  # DOWN and RIGHT
            "move_count": 0,
        }
        key_ul = encode_observation(obs_up_and_left)
        key_dr = encode_observation(obs_down_and_right)
        assert key_ul != key_dr
        assert key_ul[1] != key_dr[1]  # Masks differ


class TestAgentEncodesObservation:
    """Test QLearningAgent.encode_observation delegation."""

    def test_agent_encode_observation_delegates_correctly(self) -> None:
        """QLearningAgent.encode_observation delegates to state_encoding."""
        agent = QLearningAgent(Role.COP)
        obs = {
            "self_position": [1, 1],
            "opponent_position": [2, 2],
            "barriers": [],
            "move_count": 0,
        }
        key = agent.encode_observation(obs)
        assert key == ((1, 1), 0)  # (dr, dc), barrier_mask


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


class TestPersistenceRoundTrip:
    """Persistence round-trip: save() and load() reproduce identical q_table."""

    def test_save_and_load_empty_table(self, tmp_path: Path) -> None:
        """Saving and loading an empty q_table works."""
        agent1 = QLearningAgent(Role.COP)
        save_path = tmp_path / "q_empty.json"
        agent1.save(save_path)

        agent2 = QLearningAgent(Role.COP)
        agent2.load(save_path)

        assert agent2.q_table == {}

    def test_save_and_load_single_state_single_action(self, tmp_path: Path) -> None:
        """Saving and loading a single state with one modified action."""
        agent1 = QLearningAgent(Role.COP)
        state_key: StateKey = ((1, 0), 0)
        agent1.update(state_key, Action.UP, 5.0, ((0, 1), 0), done=False)

        save_path = tmp_path / "q_single.json"
        agent1.save(save_path)

        agent2 = QLearningAgent(Role.COP)
        agent2.load(save_path)

        assert agent2.q_table == agent1.q_table
        q1_up = agent1.q_table[state_key][Action.UP]
        q2_up = agent2.q_table[state_key][Action.UP]
        assert q2_up == q1_up

    def test_save_and_load_multiple_states_mixed_opponents(
        self, tmp_path: Path
    ) -> None:
        """Save/load with varied state keys (some None opponent, some with deltas)."""
        agent1 = QLearningAgent(Role.COP)

        # State with visible opponent
        state1: StateKey = ((1, 0), 0)
        agent1.update(state1, Action.UP, 5.0, ((0, 1), 0), done=False)

        # State with unseen opponent (None)
        state2: StateKey = (None, 3)
        agent1.update(state2, Action.DOWN, 3.0, ((2, 2), 1), done=False)

        # Another state with different barrier mask
        state3: StateKey = ((-1, 1), 5)
        agent1.update(state3, Action.LEFT, 7.0, ((0, 0), 2), done=False)

        save_path = tmp_path / "q_mixed.json"
        agent1.save(save_path)

        agent2 = QLearningAgent(Role.COP)
        agent2.load(save_path)

        assert agent2.q_table == agent1.q_table
        assert state2 in agent2.q_table  # None-opponent key preserved
        q1_down = agent1.q_table[state2][Action.DOWN]
        q2_down = agent2.q_table[state2][Action.DOWN]
        assert q2_down == q1_down

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        """save() creates parent directories if they don't exist."""
        agent = QLearningAgent(Role.COP)
        agent.update(((1, 0), 0), Action.UP, 5.0, ((0, 1), 0), done=False)

        nested_path = tmp_path / "nested" / "deeply" / "q.json"
        agent.save(nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_load_nonexistent_path_returns_empty(self, tmp_path: Path) -> None:
        """load() on a nonexistent path leaves q_table as {}."""
        agent = QLearningAgent(Role.COP)
        nonexistent = tmp_path / "does_not_exist.json"

        agent.load(nonexistent)

        assert agent.q_table == {}

    def test_load_nonexistent_doesnt_raise(self, tmp_path: Path) -> None:
        """load() does not raise an exception when path does not exist."""
        agent = QLearningAgent(Role.COP)
        nonexistent = tmp_path / "nonexistent" / "path" / "q.json"

        # Should not raise
        agent.load(nonexistent)
        assert agent.q_table == {}

    def test_round_trip_preserves_all_values_exactly(self, tmp_path: Path) -> None:
        """Round-trip preserves all state keys and values."""
        agent1 = QLearningAgent(Role.COP, alpha=0.15, gamma=0.85)

        # Populate with varied updates
        updates = [
            (((1, 0), 0), Action.UP, 5.0, ((0, 1), 0), False),
            ((None, 2), Action.DOWN, -3.5, ((1, 1), 1), False),
            (((2, -1), 7), Action.LEFT, 0.0, ((1, 0), 3), True),
            (((0, 2), 15), Action.RIGHT, 10.5, ((-1, 2), 4), False),
        ]

        for state, action, reward, next_state, done in updates:
            agent1.update(state, action, reward, next_state, done)

        save_path = tmp_path / "q_full.json"
        agent1.save(save_path)

        agent2 = QLearningAgent(Role.COP)
        agent2.load(save_path)

        # Deep comparison: same keys, same action dicts, same float values
        assert set(agent2.q_table.keys()) == set(agent1.q_table.keys())
        for state_key in agent1.q_table:
            a1_actions = set(agent1.q_table[state_key].keys())
            a2_actions = set(agent2.q_table[state_key].keys())
            assert a2_actions == a1_actions
            for action in agent1.q_table[state_key]:
                q1 = agent1.q_table[state_key][action]
                q2 = agent2.q_table[state_key][action]
                assert q2 == q1

    def test_persisted_json_structure_readable(self, tmp_path: Path) -> None:
        """Persisted JSON file is readable and has expected structure."""
        agent = QLearningAgent(Role.COP)
        agent.update(((1, 0), 0), Action.UP, 5.0, ((0, 1), 0), done=False)
        agent.update((None, 3), Action.DOWN, 3.0, ((0, 0), 0), done=False)

        save_path = tmp_path / "q_structure.json"
        agent.save(save_path)

        # Read raw JSON
        raw = json.loads(save_path.read_text())
        assert isinstance(raw, dict)
        # Keys are JSON strings (from json.dumps of StateKey)
        for key_str, action_dict in raw.items():
            assert isinstance(key_str, str)
            assert isinstance(action_dict, dict)
            # Action values are stored as strings (Action.value)
            for action_str, q_value in action_dict.items():
                assert isinstance(action_str, str)
                assert isinstance(q_value, (int, float))

    def test_encode_decode_state_key_roundtrip(self) -> None:
        """_encode_state_key and _decode_state_key round-trip correctly."""
        test_keys = [
            ((1, 0), 0),
            (None, 15),
            ((-2, 3), 7),
            ((0, 0), 0),
        ]
        for key in test_keys:
            encoded = _encode_state_key(key)
            decoded = _decode_state_key(encoded)
            assert decoded == key

    def test_encode_decode_values_roundtrip(self) -> None:
        """_encode_values and _decode_values round-trip correctly."""
        test_values = {
            Action.UP: 1.5,
            Action.DOWN: -2.0,
            Action.LEFT: 0.0,
            Action.RIGHT: 10.5,
            Action.STAY: 3.14159,
        }
        encoded = _encode_values(test_values)
        decoded = _decode_values(encoded)
        assert decoded == test_values

    def test_save_q_table_load_q_table_functions(self, tmp_path: Path) -> None:
        """Test save_q_table and load_q_table functions directly."""
        q_table1: dict = {
            ((1, 0), 0): {
                Action.UP: 5.0,
                Action.DOWN: 0.0,
                Action.LEFT: 0.0,
                Action.RIGHT: 0.0,
                Action.STAY: 0.0,
            },
            (None, 3): {
                Action.UP: 0.0,
                Action.DOWN: 3.0,
                Action.LEFT: 0.0,
                Action.RIGHT: 0.0,
                Action.STAY: 0.0,
            },
        }

        save_path = tmp_path / "q_direct.json"
        save_q_table(save_path, q_table1)
        q_table2 = load_q_table(save_path)

        assert q_table2 == q_table1


class TestStateEncodingLocalityAndBoundedness:
    """Additional tests for barrier adjacency locality and generalization."""

    def test_barrier_at_boundary_no_wrap_around(self) -> None:
        """Barriers at grid boundary don't wrap around."""
        # Agent at top-left corner with barrier at expected UP position (out of bounds)
        obs = {
            "self_position": [0, 0],
            "opponent_position": None,
            "barriers": [],  # No barriers listed
            "move_count": 0,
        }
        key = encode_observation(obs)
        # No UP barrier should exist (it would be at [-1, 0] which doesn't exist)
        assert key[1] == 0  # Empty mask

    def test_all_four_barriers_adjacent(self) -> None:
        """All four adjacent barriers produce mask = 0b1111 = 15."""
        obs = {
            "self_position": [2, 2],
            "opponent_position": None,
            "barriers": [
                [1, 2],  # UP
                [3, 2],  # DOWN
                [2, 1],  # LEFT
                [2, 3],  # RIGHT
            ],
            "move_count": 0,
        }
        key = encode_observation(obs)
        assert key[1] == 15  # 0b1111

    def test_only_up_barrier(self) -> None:
        """Only UP barrier produces mask = 0b0001 = 1."""
        obs = {
            "self_position": [2, 2],
            "opponent_position": None,
            "barriers": [[1, 2]],
            "move_count": 0,
        }
        key = encode_observation(obs)
        assert key[1] == 1

    def test_only_down_barrier(self) -> None:
        """Only DOWN barrier produces mask = 0b0010 = 2."""
        obs = {
            "self_position": [2, 2],
            "opponent_position": None,
            "barriers": [[3, 2]],
            "move_count": 0,
        }
        key = encode_observation(obs)
        assert key[1] == 2

    def test_only_left_barrier(self) -> None:
        """Only LEFT barrier produces mask = 0b0100 = 4."""
        obs = {
            "self_position": [2, 2],
            "opponent_position": None,
            "barriers": [[2, 1]],
            "move_count": 0,
        }
        key = encode_observation(obs)
        assert key[1] == 4

    def test_only_right_barrier(self) -> None:
        """Only RIGHT barrier produces mask = 0b1000 = 8."""
        obs = {
            "self_position": [2, 2],
            "opponent_position": None,
            "barriers": [[2, 3]],
            "move_count": 0,
        }
        key = encode_observation(obs)
        assert key[1] == 8
