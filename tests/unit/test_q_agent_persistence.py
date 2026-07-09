"""Persistence round-trip tests for Q-learning agent (part 1: save/load basics)."""

from __future__ import annotations

from pathlib import Path

from agents.q_agent import QLearningAgent
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
