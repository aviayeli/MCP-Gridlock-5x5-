"""Tests for MatchState.observation_for() and .status()."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Action, Role
from mcp_server.match_state import MatchState


class TestMatchStateObservation:
    """Test MatchState.observation_for() returns correct fog-of-war view."""

    def test_observation_before_any_move(self) -> None:
        """Observation reflects initial positions even before first turn."""
        match = MatchState()
        cop_obs = match.observation_for(Role.COP)
        thief_obs = match.observation_for(Role.THIEF)

        # Cop sees itself at (0,0), Thief at (4,4) out of visibility range
        assert cop_obs.self_position == (0, 0)
        # Distance (0,0) to (4,4) is 8, exceeds VISIBILITY_RADIUS=2
        assert cop_obs.opponent_position is None

        # Thief sees itself at (4,4), Cop out of visibility range
        assert thief_obs.self_position == (4, 4)
        assert thief_obs.opponent_position is None

        # Both see barriers (empty frozenset or populated depending on board)
        assert isinstance(cop_obs.barriers, frozenset)
        assert isinstance(thief_obs.barriers, frozenset)

        # Move count is 0
        assert cop_obs.move_count == 0
        assert thief_obs.move_count == 0

    def test_observation_fog_of_war_out_of_range(self) -> None:
        """Opponent position is None when outside visibility radius."""
        # Create custom config with max_barriers=0 to make placement deterministic
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Cop at (0,0), Thief at (4,4) are far apart (manhattan distance = 8)
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        match = MatchState(game)

        cop_obs = match.observation_for(Role.COP)
        thief_obs = match.observation_for(Role.THIEF)

        # Distance is 8, visibility radius is 2, so opponent is unseen
        assert cop_obs.opponent_position is None
        assert thief_obs.opponent_position is None

    def test_observation_fog_of_war_in_range(self) -> None:
        """Opponent position is visible when within visibility radius."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Cop at (2,2), Thief at (2,4) are 2 apart (manhattan distance = 2)
        # This is exactly at VISIBILITY_RADIUS
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 4))
        match = MatchState(game)

        cop_obs = match.observation_for(Role.COP)
        thief_obs = match.observation_for(Role.THIEF)

        # Both should see the opponent
        assert cop_obs.opponent_position == (2, 4)
        assert thief_obs.opponent_position == (2, 2)

    def test_observation_updates_after_move(self) -> None:
        """Observation reflects updated positions after a move."""
        match = MatchState()
        # First move: Cop moves UP (to -1, but clamped to 0), Thief stays
        # Actually UP from (0,0) is invalid, so Cop stays at (0,0)
        result = match.submit_move(Role.COP, Action.UP)
        assert result is None  # Waiting for Thief
        result = match.submit_move(Role.THIEF, Action.STAY)
        assert result is not None  # Both submitted, resolved

        # After first turn, move_count should be 1
        cop_obs = match.observation_for(Role.COP)
        assert cop_obs.move_count == 1


class TestMatchStateStatus:
    """Test MatchState.status() returns correct game state."""

    def test_status_before_any_move(self) -> None:
        """status() returns correct initial state."""
        match = MatchState()
        status = match.status()

        assert status["move_count"] == 0
        assert status["done"] is False
        assert status["cop_reward"] is None
        assert status["thief_reward"] is None

    def test_status_mid_game(self) -> None:
        """status() returns move_count but None rewards mid-game."""
        match = MatchState()
        match.submit_move(Role.COP, Action.STAY)
        match.submit_move(Role.THIEF, Action.STAY)

        status = match.status()
        assert status["move_count"] == 1
        assert status["done"] is False
        assert status["cop_reward"] is None
        assert status["thief_reward"] is None

    def test_status_after_game_end(self) -> None:
        """status() returns rewards after game ends."""
        config = GameConfig(
            rows=5, cols=5, max_moves=1, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 2))
        match = MatchState(game)

        match.submit_move(Role.COP, Action.STAY)
        match.submit_move(Role.THIEF, Action.STAY)

        status = match.status()
        assert status["done"] is True
        assert status["cop_reward"] == 20  # cop_win for capture
        assert status["thief_reward"] == 5  # thief_loss for capture
