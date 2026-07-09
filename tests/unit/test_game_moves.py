"""Tests for GameLoop: valid moves."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Action


class TestValidMoves:
    """Test valid moves update player position correctly."""

    def test_move_up(self) -> None:
        """Moving UP should decrease row by 1."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(4, 4))
        result = game.step(Action.UP, Action.STAY)
        assert result.cop_observation.self_position == (1, 2)
        assert game.cop.position == (1, 2)

    def test_move_down(self) -> None:
        """Moving DOWN should increase row by 1."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(4, 4))
        result = game.step(Action.DOWN, Action.STAY)
        assert result.cop_observation.self_position == (3, 2)
        assert game.cop.position == (3, 2)

    def test_move_left(self) -> None:
        """Moving LEFT should decrease column by 1."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(4, 4))
        result = game.step(Action.LEFT, Action.STAY)
        assert result.cop_observation.self_position == (2, 1)
        assert game.cop.position == (2, 1)

    def test_move_right(self) -> None:
        """Moving RIGHT should increase column by 1."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(4, 4))
        result = game.step(Action.RIGHT, Action.STAY)
        assert result.cop_observation.self_position == (2, 3)
        assert game.cop.position == (2, 3)

    def test_stay_action(self) -> None:
        """STAY action should not change position."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(4, 4))
        result = game.step(Action.STAY, Action.STAY)
        assert result.cop_observation.self_position == (2, 2)
        assert game.cop.position == (2, 2)

    def test_all_actions_for_both_players(self) -> None:
        """All 5 actions should work for both cop and thief."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        start_positions = [(1, 1), (3, 3)]
        expected_positions = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
            Action.STAY: (0, 0),
        }
        for action in expected_positions:
            game = GameLoop(
                config=config, cop_start=start_positions[0],
                thief_start=start_positions[1],
            )
            result = game.step(action, Action.STAY)
            dr, dc = expected_positions[action]
            expected_cop_pos = (start_positions[0][0] + dr, start_positions[0][1] + dc)
            assert result.cop_observation.self_position == expected_cop_pos

    def test_consecutive_valid_moves(self) -> None:
        """Multiple consecutive valid moves should chain correctly."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(4, 4))
        game.step(Action.UP, Action.STAY)
        assert game.cop.position == (1, 2)
        game.step(Action.UP, Action.STAY)
        assert game.cop.position == (0, 2)
        game.step(Action.RIGHT, Action.STAY)
        assert game.cop.position == (0, 3)
