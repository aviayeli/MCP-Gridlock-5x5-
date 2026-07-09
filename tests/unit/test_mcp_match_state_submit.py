"""Tests for MatchState.submit_move() buffering and resolution."""

from __future__ import annotations

import pytest

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Action, Role
from mcp_server.match_state import MatchState


class TestMatchStateSubmitMove:
    """Test MatchState.submit_move() buffering and resolution."""

    def test_submit_move_returns_none_waiting_for_other_role(self) -> None:
        """submit_move returns None while waiting for the other role."""
        match = MatchState()
        result = match.submit_move(Role.COP, Action.DOWN)
        assert result is None

    def test_submit_move_returns_turn_result_when_both_submitted(self) -> None:
        """submit_move returns TurnResult once both roles have submitted."""
        match = MatchState()
        result_cop = match.submit_move(Role.COP, Action.DOWN)
        assert result_cop is None

        result_thief = match.submit_move(Role.THIEF, Action.RIGHT)
        assert result_thief is not None
        # Verify it's a TurnResult-like object with expected fields
        assert hasattr(result_thief, "move_count")
        assert hasattr(result_thief, "done")
        assert hasattr(result_thief, "cop_observation")
        assert hasattr(result_thief, "thief_observation")
        assert result_thief.move_count == 1

    def test_buffer_clears_for_next_turn(self) -> None:
        """Buffer clears after resolving a turn, allowing turn 2."""
        match = MatchState()

        # Turn 1
        result1 = match.submit_move(Role.COP, Action.DOWN)
        assert result1 is None
        result1 = match.submit_move(Role.THIEF, Action.STAY)
        assert result1 is not None
        assert result1.move_count == 1

        # Turn 2: should resolve as a fresh turn, not stale state
        result2 = match.submit_move(Role.COP, Action.STAY)
        assert result2 is None  # Waiting for Thief again
        result2 = match.submit_move(Role.THIEF, Action.STAY)
        assert result2 is not None
        assert result2.move_count == 2  # Confirm move_count incremented

    def test_role_can_revise_pending_action_before_partner_arrives(self) -> None:
        """A role can overwrite its pending action before the turn resolves.

        Uses max_barriers=0 (rather than the default MatchState()) so the
        Cop's RIGHT move is deterministically legal — with random barrier
        placement enabled, a barrier could occasionally land on (0, 1) and
        make this assertion flaky.
        """
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        match = MatchState(GameLoop(config=config))
        result1 = match.submit_move(Role.COP, Action.DOWN)
        assert result1 is None

        # Cop revises its move from DOWN to RIGHT
        result_revised = match.submit_move(Role.COP, Action.RIGHT)
        assert result_revised is None  # Still waiting

        # Thief submits; the revised Cop action (RIGHT) should be used
        result = match.submit_move(Role.THIEF, Action.STAY)
        assert result is not None
        # Cop starts at (0,0), moves RIGHT -> (0,1)
        assert result.cop_observation.self_position == (0, 1)

    def test_submit_move_raises_runtime_error_after_done(self) -> None:
        """submit_move raises RuntimeError after the match is done."""
        config = GameConfig(
            rows=5, cols=5, max_moves=1, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Start close together to guarantee capture on first turn
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 2))
        match = MatchState(game)

        # First move: they're at same position, so game ends
        result = match.submit_move(Role.COP, Action.STAY)
        assert result is None
        result = match.submit_move(Role.THIEF, Action.STAY)
        assert result is not None
        assert result.done is True

        # Attempt to submit a move after done should raise RuntimeError
        with pytest.raises(RuntimeError, match="cannot submit a move"):
            match.submit_move(Role.COP, Action.STAY)
