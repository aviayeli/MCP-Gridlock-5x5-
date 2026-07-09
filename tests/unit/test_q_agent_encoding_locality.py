"""Agent observation encoding and barrier locality tests."""

from __future__ import annotations

from agents.q_agent import QLearningAgent
from agents.state_encoding import encode_observation
from engine.player import Role


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
