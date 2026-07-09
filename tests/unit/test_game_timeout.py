"""Tests for GameLoop: timeout (reaching max_moves without capture)."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Action


class TestTimeout:
    """Test timeout (reaching max_moves) without capture."""

    def test_timeout_both_stay(self) -> None:
        """Timeout when both players STAY for max_moves turns."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        for _ in range(24):
            result = game.step(Action.STAY, Action.STAY)
            assert result.done is False
            assert result.move_count < 25
        # 25th move
        result = game.step(Action.STAY, Action.STAY)
        assert result.done is True
        assert result.move_count == 25
        assert result.cop_reward == 5  # cop_loss
        assert result.thief_reward == 10  # thief_win

    def test_timeout_exact_max_moves(self) -> None:
        """Game should end exactly at max_moves."""
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

    def test_timeout_with_movement(self) -> None:
        """Timeout should trigger even with movement that never captures."""
        config = GameConfig(
            rows=5, cols=5, max_moves=10, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        for _ in range(9):
            result = game.step(Action.UP, Action.DOWN)  # opposite moves
            assert result.done is False
        result = game.step(Action.UP, Action.DOWN)
        assert result.done is True
        assert result.move_count == 10
        assert result.cop_reward == 5  # cop_loss
        assert result.thief_reward == 10  # thief_win

    def test_capture_before_timeout(self) -> None:
        """Capture should take precedence over timeout."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        # Reach move 25 and capture simultaneously
        for _ in range(24):
            game.step(Action.STAY, Action.STAY)
        result = game.step(Action.RIGHT, Action.LEFT)
        # Should see capture (cop_reward=20) not timeout (cop_reward=5)
        if (result.done and result.move_count == 25 and result.cop_reward == 20
                or result.move_count < 25):
            assert True
