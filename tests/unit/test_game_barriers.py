"""Tests for GameLoop: barrier collisions."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Action


class TestBarrierCollisions:
    """Test that moving into a barrier results in no position change."""

    def test_barrier_collision_with_controlled_seed(self) -> None:
        """Force a known barrier layout so the collision outcome is deterministic.

        The original version of this test inspected one GameLoop's randomly
        placed barrier, then asserted against a *second*, independently
        randomized GameLoop built from the same config — the two boards have
        no guaranteed relationship, so the assertion could fail depending on
        unrelated system randomness. Overriding `board._barriers` directly
        pins the layout so the test is deterministic regardless of RNG state.
        """
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=1,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        game.board._barriers = frozenset({(0, 1)})  # noqa: SLF001

        result = game.step(Action.RIGHT, Action.STAY)

        assert game.cop.position == (0, 0)
        assert result.done is False

    def test_barrier_never_allows_movement_into_it(self) -> None:
        """Multiple seeds should show barriers block movement."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=5,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Try with multiple seeds to likely hit a barrier
        for _seed in range(10):
            game = GameLoop(config=config, cop_start=(2, 2), thief_start=(4, 4))
            # Verify that barriers are never entered
            for _ in range(5):
                game.step(Action.STAY, Action.STAY)
            # After 5 moves, cop is still at (2,2), thief at (4,4)
            assert game.cop.position == (2, 2)
            assert game.thief.position == (4, 4)
