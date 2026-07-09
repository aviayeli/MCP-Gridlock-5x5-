"""Tests for GameLoop: state management and turn results."""

from __future__ import annotations

import pytest

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.observation import Observation
from engine.player import Action


class TestGameLoopStateManagement:
    """Test GameLoop state tracking and terminal conditions."""

    def test_move_count_increments(self) -> None:
        """move_count should increment on each step."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        for i in range(5):
            result = game.step(Action.STAY, Action.STAY)
            assert game.move_count == i + 1
            assert result.move_count == i + 1

    def test_done_false_initially(self) -> None:
        """done should be False at game start."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        assert game.done is False

    def test_done_true_after_capture(self) -> None:
        """done should be True after capture."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 2))
        result = game.step(Action.STAY, Action.STAY)
        assert game.done is True
        assert result.done is True

    def test_step_after_done_raises_error(self) -> None:
        """Calling step() after done should raise RuntimeError."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 2))
        game.step(Action.STAY, Action.STAY)
        assert game.done is True
        msg = "step\\(\\) called after episode already ended"
        with pytest.raises(RuntimeError, match=msg):
            game.step(Action.STAY, Action.STAY)

    def test_step_after_timeout_raises_error(self) -> None:
        """Calling step() after timeout should raise RuntimeError."""
        config = GameConfig(
            rows=5, cols=5, max_moves=3, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        for _ in range(3):
            game.step(Action.STAY, Action.STAY)
        assert game.done is True
        msg = "step\\(\\) called after episode already ended"
        with pytest.raises(RuntimeError, match=msg):
            game.step(Action.STAY, Action.STAY)


class TestTurnResult:
    """Test TurnResult data structure and content."""

    def test_turn_result_contains_observations(self) -> None:
        """TurnResult should contain both cop and thief observations."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        result = game.step(Action.STAY, Action.STAY)
        assert isinstance(result.cop_observation, Observation)
        assert isinstance(result.thief_observation, Observation)

    def test_turn_result_move_count_matches_game(self) -> None:
        """TurnResult.move_count should match GameLoop.move_count."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        for _ in range(5):
            result = game.step(Action.STAY, Action.STAY)
            assert result.move_count == game.move_count

    def test_turn_result_immutable(self) -> None:
        """TurnResult should be immutable (frozen dataclass)."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        result = game.step(Action.STAY, Action.STAY)
        with pytest.raises(AttributeError):
            result.done = True  # type: ignore[misc]
