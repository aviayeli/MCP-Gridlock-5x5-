"""Persistence round-trip tests for Q-learning agent."""

from __future__ import annotations

import json
from pathlib import Path

from agents.q_agent import QLearningAgent
from agents.q_persistence import (
    _decode_state_key,
    _decode_values,
    _encode_state_key,
    _encode_values,
    load_q_table,
    save_q_table,
)
from agents.state_encoding import StateKey
from engine.player import Action, Role


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
