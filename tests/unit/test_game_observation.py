"""Tests for GameLoop: fog-of-war observation hides/reveals opponent position."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.observation import VISIBILITY_RADIUS, manhattan_distance
from engine.player import Action


class TestFogOfWar:
    """Test fog-of-war observation hides/reveals opponent position."""

    def test_opponent_visible_within_radius(self) -> None:
        """Opponent position revealed when distance <= VISIBILITY_RADIUS."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 4))
        # Distance is 2 (Manhattan) = VISIBILITY_RADIUS
        result = game.step(Action.STAY, Action.STAY)
        assert result.cop_observation.opponent_position == (2, 4)

    def test_opponent_hidden_outside_radius(self) -> None:
        """Opponent position hidden when distance > VISIBILITY_RADIUS."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        # Distance is 8 (Manhattan) > VISIBILITY_RADIUS (2)
        result = game.step(Action.STAY, Action.STAY)
        assert result.cop_observation.opponent_position is None

    def test_visibility_radius_boundary(self) -> None:
        """Test boundary: distance = VISIBILITY_RADIUS."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(2, 0))
        # Distance is exactly 2
        result = game.step(Action.STAY, Action.STAY)
        assert result.cop_observation.opponent_position == (2, 0)

    def test_visibility_radius_just_outside(self) -> None:
        """Test boundary: distance = VISIBILITY_RADIUS + 1."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(3, 0))
        # Distance is 3
        result = game.step(Action.STAY, Action.STAY)
        assert result.cop_observation.opponent_position is None

    def test_both_observations_symmetric(self) -> None:
        """Both cop and thief should have same visibility symmetry."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(4, 4))
        result = game.step(Action.STAY, Action.STAY)
        # Both should have same visibility since manhattan distance is symmetric
        distance = manhattan_distance((2, 2), (4, 4))
        visible = distance <= VISIBILITY_RADIUS
        assert (result.cop_observation.opponent_position is not None) == visible
        assert (result.thief_observation.opponent_position is not None) == visible

    def test_self_position_always_visible(self) -> None:
        """Own position should always be visible."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        result = game.step(Action.STAY, Action.STAY)
        assert result.cop_observation.self_position == (0, 0)
        assert result.thief_observation.self_position == (4, 4)

    def test_barriers_always_visible(self) -> None:
        """Barriers should always be visible."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=5,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        result = game.step(Action.STAY, Action.STAY)
        barriers = game.board.barrier_coordinates()
        assert len(result.cop_observation.barriers) == len(barriers)
        assert result.cop_observation.barriers == barriers

    def test_move_count_always_visible(self) -> None:
        """Move count should always be visible."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        for i in range(5):
            result = game.step(Action.STAY, Action.STAY)
            assert result.cop_observation.move_count == i + 1
            assert result.thief_observation.move_count == i + 1
