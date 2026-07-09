"""Test results JSON schema and structure validation."""

from __future__ import annotations

import json
from pathlib import Path

from engine.config_loader import load_config
from engine.tournament_report import Team
from scripts.run_tournament import main


class TestResultsJsonSchema:
    """Test that main() produces correctly-structured results JSON."""

    def test_main_generates_results_json(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """main() generates a valid tournament_report.json file."""
        results_file = tmp_path / "tournament_report.json"
        data_dir = tmp_path / "data"
        q_tables = {
            Team.ALPHA: data_dir / "q_table_team_alpha.json",
            Team.BETA: data_dir / "q_table_team_beta.json",
        }

        monkeypatch.setattr(
            "scripts.run_tournament.RESULTS_PATH", results_file
        )
        monkeypatch.setattr("scripts.run_tournament.DATA_DIR", data_dir)
        monkeypatch.setattr("scripts.run_tournament.Q_TABLE_PATHS", q_tables)

        main()

        assert results_file.exists(), "tournament_report.json not created"
        data = json.loads(results_file.read_text())
        assert isinstance(data, dict), "Results JSON is not a dict"

    def test_main_results_schema_valid(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """main() produces a results JSON with correct top-level schema."""
        results_file = tmp_path / "tournament_report.json"
        data_dir = tmp_path / "data"
        q_tables = {
            Team.ALPHA: data_dir / "q_table_team_alpha.json",
            Team.BETA: data_dir / "q_table_team_beta.json",
        }

        monkeypatch.setattr(
            "scripts.run_tournament.RESULTS_PATH", results_file
        )
        monkeypatch.setattr("scripts.run_tournament.DATA_DIR", data_dir)
        monkeypatch.setattr("scripts.run_tournament.Q_TABLE_PATHS", q_tables)

        main()

        data = json.loads(results_file.read_text())

        assert "num_games" in data
        assert "games" in data
        assert "final_scores" in data
        assert "winner" in data

        assert isinstance(data["num_games"], int)
        assert isinstance(data["games"], list)
        assert isinstance(data["final_scores"], dict)
        assert isinstance(data["winner"], str)

    def test_main_results_num_games_matches_config(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """main() sets num_games from config."""
        results_file = tmp_path / "tournament_report.json"
        data_dir = tmp_path / "data"
        q_tables = {
            Team.ALPHA: data_dir / "q_table_team_alpha.json",
            Team.BETA: data_dir / "q_table_team_beta.json",
        }

        monkeypatch.setattr(
            "scripts.run_tournament.RESULTS_PATH", results_file
        )
        monkeypatch.setattr("scripts.run_tournament.DATA_DIR", data_dir)
        monkeypatch.setattr("scripts.run_tournament.Q_TABLE_PATHS", q_tables)

        config = load_config()
        main()

        data = json.loads(results_file.read_text())
        assert data["num_games"] == config.num_games

    def test_main_results_games_count_matches_num_games(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """main() creates one game entry per num_games."""
        results_file = tmp_path / "tournament_report.json"
        data_dir = tmp_path / "data"
        q_tables = {
            Team.ALPHA: data_dir / "q_table_team_alpha.json",
            Team.BETA: data_dir / "q_table_team_beta.json",
        }

        monkeypatch.setattr(
            "scripts.run_tournament.RESULTS_PATH", results_file
        )
        monkeypatch.setattr("scripts.run_tournament.DATA_DIR", data_dir)
        monkeypatch.setattr("scripts.run_tournament.Q_TABLE_PATHS", q_tables)

        config = load_config()
        main()

        data = json.loads(results_file.read_text())
        assert len(data["games"]) == config.num_games
