"""State encoding determinism tests for Q-learning agent: barrier masking (part 2)."""

from __future__ import annotations

from agents.state_encoding import encode_observation


class TestStateEncodingBarrierMasking:
    """Barrier-adjacency mask locality and composition, plus move_count exclusion."""

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
