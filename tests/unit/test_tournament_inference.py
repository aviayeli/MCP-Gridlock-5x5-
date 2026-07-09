"""Tests for game outcome inference (CAPTURE vs TIMEOUT)."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.player import Action
from engine.tournament import Tournament
from engine.tournament_report import Outcome, Team


class TestTournamentOutcomeInference:
    """Test that game outcomes (CAPTURE vs TIMEOUT) are correctly inferred."""

    def test_outcome_capture_when_cop_win_reward(self) -> None:
        """Outcome is CAPTURE when cop_reward == config.scoring.cop_win."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Place them adjacent so collision is immediate
        policies = {
            Team.ALPHA: lambda obs: Action.DOWN,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(
            policies, config=config, cop_start=(0, 0), thief_start=(1, 0)
        )
        tournament.play()

        record = tournament.records[0]
        # With ALPHA as Cop starting at (0,0) and THIEF starting at (1,0),
        # ALPHA moving DOWN reaches (1,0), capturing the THIEF
        if record.cop_team == Team.ALPHA:
            # Cop should get cop_win
            assert record.cop_reward == 20
            assert record.outcome == Outcome.CAPTURE

    def test_outcome_timeout_when_cop_loss_reward(self) -> None:
        """Outcome is TIMEOUT when cop_reward == config.scoring.cop_loss."""
        config = GameConfig(
            rows=5, cols=5, max_moves=2, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Place them far apart so no collision
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(
            policies, config=config, cop_start=(0, 0), thief_start=(4, 4)
        )
        tournament.play()

        record = tournament.records[0]
        # With max_moves=2 and both staying, timeout occurs
        assert record.cop_reward == 5
        assert record.outcome == Outcome.TIMEOUT
