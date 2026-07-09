"""Tests for Tournament.report() schema and structure."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.player import Action
from engine.tournament import Tournament
from engine.tournament_report import Team


class TestTournamentReport:
    """Test tournament.report() and report schema."""

    def test_report_schema_has_required_keys(self) -> None:
        """Report has num_games, games, final_scores, and winner."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        report = tournament.report()
        assert "num_games" in report
        assert "games" in report
        assert "final_scores" in report
        assert "winner" in report

    def test_report_num_games_matches_config(self) -> None:
        """Report num_games matches config.num_games."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=4, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        report = tournament.report()
        assert report["num_games"] == 4

    def test_report_games_list_length_matches_num_games(self) -> None:
        """Report games list has one entry per game."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=3, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        report = tournament.report()
        assert len(report["games"]) == 3

    def test_report_final_scores_keys_are_team_names(self) -> None:
        """Report final_scores keys are 'ALPHA' and 'BETA'."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        report = tournament.report()
        assert "ALPHA" in report["final_scores"]
        assert "BETA" in report["final_scores"]
        assert len(report["final_scores"]) == 2

    def test_report_games_have_correct_structure(self) -> None:
        """Each game record in report has expected fields."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        report = tournament.report()
        game = report["games"][0]
        assert "game_index" in game
        assert "cop_team" in game
        assert "thief_team" in game
        assert "moves" in game
        assert "cop_reward" in game
        assert "thief_reward" in game
        assert "outcome" in game
