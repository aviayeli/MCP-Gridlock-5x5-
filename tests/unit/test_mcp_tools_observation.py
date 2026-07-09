"""Tests for GameTools.get_observation()."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Role
from mcp_server.match_state import MatchState
from mcp_server.tools import GameTools


class TestGameToolsGetObservation:
    """Test GameTools.get_observation() returns correct dict shape."""

    def test_get_observation_returns_dict_with_correct_shape(self) -> None:
        """get_observation returns a dict with expected keys."""
        match = MatchState()
        tools = GameTools(match)

        cop_obs = tools.get_observation(Role.COP)
        assert isinstance(cop_obs, dict)
        assert "self_position" in cop_obs
        assert "opponent_position" in cop_obs
        assert "barriers" in cop_obs
        assert "move_count" in cop_obs

    def test_get_observation_serializes_positions_as_lists(self) -> None:
        """Positions are lists, not tuples (or None if out of range)."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Place them 2 apart (exactly at VISIBILITY_RADIUS)
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 4))
        tools = GameTools(MatchState(game))

        cop_obs = tools.get_observation(Role.COP)
        assert isinstance(cop_obs["self_position"], list)
        assert cop_obs["self_position"] == [2, 2]
        assert isinstance(cop_obs["opponent_position"], list)
        assert cop_obs["opponent_position"] == [2, 4]

    def test_get_observation_serializes_barriers_as_list_of_lists(self) -> None:
        """Barriers are a list of [row, col] lists."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=5,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config)
        tools = GameTools(MatchState(game))

        obs = tools.get_observation(Role.COP)
        assert isinstance(obs["barriers"], list)
        # Each barrier should be a list of 2 integers
        for barrier in obs["barriers"]:
            assert isinstance(barrier, list)
            assert len(barrier) == 2
            assert isinstance(barrier[0], int)
            assert isinstance(barrier[1], int)

    def test_get_observation_fog_of_war(self) -> None:
        """get_observation respects fog-of-war (opponent_position None when unseen)."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        tools = GameTools(MatchState(game))

        cop_obs = tools.get_observation(Role.COP)
        # Distance 8 > VISIBILITY_RADIUS, so opponent_position should be None
        assert cop_obs["opponent_position"] is None

    def test_get_observation_fog_of_war_in_range(self) -> None:
        """get_observation reveals opponent when in range."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 4))
        tools = GameTools(MatchState(game))

        cop_obs = tools.get_observation(Role.COP)
        # Distance 2 == VISIBILITY_RADIUS, should see opponent
        assert cop_obs["opponent_position"] == [2, 4]
