"""Tests for GameLoop: invalid moves (out of bounds)."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Action


class TestInvalidMoves:
    """Test invalid moves resolve to STAY behavior (no position change)."""

    def test_move_up_from_top_edge(self) -> None:
        """Moving UP from row 0 should resolve to STAY."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 2), thief_start=(4, 4))
        initial_pos = game.cop.position
        result = game.step(Action.UP, Action.STAY)
        assert game.cop.position == initial_pos
        assert result.cop_observation.self_position == initial_pos

    def test_move_down_from_bottom_edge(self) -> None:
        """Moving DOWN from row 4 should resolve to STAY."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(4, 2), thief_start=(0, 0))
        initial_pos = game.cop.position
        game.step(Action.DOWN, Action.STAY)
        assert game.cop.position == initial_pos

    def test_move_left_from_left_edge(self) -> None:
        """Moving LEFT from col 0 should resolve to STAY."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 0), thief_start=(4, 4))
        initial_pos = game.cop.position
        game.step(Action.LEFT, Action.STAY)
        assert game.cop.position == initial_pos

    def test_move_right_from_right_edge(self) -> None:
        """Moving RIGHT from col 4 should resolve to STAY."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 4), thief_start=(4, 0))
        initial_pos = game.cop.position
        game.step(Action.RIGHT, Action.STAY)
        assert game.cop.position == initial_pos

    def test_corner_edge_cases(self) -> None:
        """Test all four corners for out-of-bounds moves."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
        bad_moves = {
            (0, 0): [Action.UP, Action.LEFT],
            (0, 4): [Action.UP, Action.RIGHT],
            (4, 0): [Action.DOWN, Action.LEFT],
            (4, 4): [Action.DOWN, Action.RIGHT],
        }
        for corner in corners:
            for bad_move in bad_moves[corner]:
                game = GameLoop(config=config, cop_start=corner, thief_start=(2, 2))
                initial_pos = game.cop.position
                game.step(bad_move, Action.STAY)
                assert game.cop.position == initial_pos
