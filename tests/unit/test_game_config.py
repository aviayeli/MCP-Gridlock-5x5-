"""Tests for GameLoop: configuration and custom settings."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Action


class TestGameLoopConfiguration:
    """Test GameLoop with custom configurations."""

    def test_custom_grid_size(self) -> None:
        """GameLoop should work with custom grid sizes."""
        config = GameConfig(
            rows=3, cols=7, max_moves=20, num_games=1, max_barriers=2,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(2, 6))
        assert game.board.rows == 3
        assert game.board.cols == 7

    def test_custom_max_moves(self) -> None:
        """GameLoop should respect custom max_moves."""
        config = GameConfig(
            rows=5, cols=5, max_moves=10, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        for _ in range(9):
            result = game.step(Action.STAY, Action.STAY)
            assert result.done is False
        result = game.step(Action.STAY, Action.STAY)
        assert result.done is True
        assert result.move_count == 10

    def test_custom_scoring(self) -> None:
        """GameLoop should apply custom scoring."""
        scoring = Scoring(cop_win=100, thief_win=50, cop_loss=1, thief_loss=2)
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=scoring,
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 2))
        result = game.step(Action.STAY, Action.STAY)
        assert result.cop_reward == 100
        assert result.thief_reward == 2

    def test_default_config_loading(self) -> None:
        """GameLoop with no config should load default."""
        game = GameLoop()
        assert game.config.rows == 5
        assert game.config.cols == 5
        assert game.config.max_moves == 25
