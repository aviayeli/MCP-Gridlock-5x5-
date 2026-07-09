"""Tests for GameLoop: capture (cop catches thief)."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Action


class TestCapture:
    """Test capture (cop and thief on same cell) ends game with correct rewards."""

    def test_capture_same_starting_position(self) -> None:
        """Immediate capture when starting at same position."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 2))
        result = game.step(Action.STAY, Action.STAY)
        assert result.done is True
        assert result.cop_reward == 20  # cop_win
        assert result.thief_reward == 5  # thief_loss
        assert game.done is True

    def test_capture_cop_moves_to_thief(self) -> None:
        """Capture when cop moves to thief's position."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(3, 2))
        result = game.step(Action.DOWN, Action.STAY)
        assert result.done is True
        assert result.cop_reward == 20
        assert result.thief_reward == 5

    def test_capture_thief_moves_to_cop(self) -> None:
        """Capture when thief moves to cop's position."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(3, 2))
        result = game.step(Action.STAY, Action.UP)
        assert result.done is True
        assert result.cop_reward == 20
        assert result.thief_reward == 5

    def test_capture_both_move_to_same_cell(self) -> None:
        """Capture when both move to the same cell simultaneously."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 4))
        result = game.step(Action.RIGHT, Action.LEFT)
        assert result.done is True
        assert result.cop_reward == 20
        assert result.thief_reward == 5

    def test_no_reward_before_capture(self) -> None:
        """Non-capturing moves should have None rewards."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        result = game.step(Action.STAY, Action.STAY)
        assert result.done is False
        assert result.cop_reward is None
        assert result.thief_reward is None

    def test_multi_step_chase_to_capture(self) -> None:
        """Cop should catch thief after multiple moves."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(2, 0))
        # Move cop down twice to reach thief
        result = game.step(Action.DOWN, Action.STAY)
        assert result.done is False
        result = game.step(Action.DOWN, Action.STAY)
        assert result.done is True
        assert result.cop_reward == 20
        assert result.thief_reward == 5
