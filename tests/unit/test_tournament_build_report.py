"""Tests for build_report function."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.tournament_report import GameRecord, Outcome, Team, build_report


class TestBuildReport:
    """Test build_report function."""

    def test_build_report_assembles_correct_schema(self) -> None:
        """build_report returns dict with num_games, games, final_scores, winner."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=2, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        records = [
            GameRecord(
                game_index=0,
                cop_team=Team.ALPHA,
                thief_team=Team.BETA,
                moves=10,
                cop_reward=20,
                thief_reward=5,
                outcome=Outcome.CAPTURE,
            ),
            GameRecord(
                game_index=1,
                cop_team=Team.BETA,
                thief_team=Team.ALPHA,
                moves=25,
                cop_reward=5,
                thief_reward=10,
                outcome=Outcome.TIMEOUT,
            ),
        ]
        team_scores = {Team.ALPHA: 25, Team.BETA: 20}

        report = build_report(config, records, team_scores)

        assert report["num_games"] == 2
        assert len(report["games"]) == 2
        assert report["final_scores"]["ALPHA"] == 25
        assert report["final_scores"]["BETA"] == 20

    def test_build_report_detects_alpha_winner(self) -> None:
        """build_report shows ALPHA as winner when ALPHA score > BETA."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        records = []
        team_scores = {Team.ALPHA: 30, Team.BETA: 10}

        report = build_report(config, records, team_scores)
        assert report["winner"] == "ALPHA"

    def test_build_report_detects_beta_winner(self) -> None:
        """build_report shows BETA as winner when BETA score > ALPHA."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        records = []
        team_scores = {Team.ALPHA: 10, Team.BETA: 30}

        report = build_report(config, records, team_scores)
        assert report["winner"] == "BETA"

    def test_build_report_detects_tie(self) -> None:
        """build_report shows 'tie' when scores are equal."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        records = []
        team_scores = {Team.ALPHA: 50, Team.BETA: 50}

        report = build_report(config, records, team_scores)
        assert report["winner"] == "tie"

    def test_build_report_serializes_game_records(self) -> None:
        """build_report converts GameRecord objects to dicts."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        record = GameRecord(
            game_index=0,
            cop_team=Team.ALPHA,
            thief_team=Team.BETA,
            moves=10,
            cop_reward=20,
            thief_reward=5,
            outcome=Outcome.CAPTURE,
        )
        team_scores = {Team.ALPHA: 20, Team.BETA: 5}

        report = build_report(config, [record], team_scores)
        game_dict = report["games"][0]

        assert game_dict["game_index"] == 0
        assert game_dict["cop_team"] == "ALPHA"
        assert game_dict["outcome"] == "CAPTURE"
