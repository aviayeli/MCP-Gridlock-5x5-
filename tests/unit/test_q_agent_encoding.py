"""State encoding determinism tests for Q-learning agent (part 1)."""

from __future__ import annotations

from agents.state_encoding import encode_observation


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
