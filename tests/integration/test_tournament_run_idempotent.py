"""Test idempotency: multiple independent runs produce consistent results."""

from __future__ import annotations

import json
from pathlib import Path

from engine.tournament_report import Team
from scripts.run_tournament import main


class TestIdempotentRuns:
    """Test that main() can be called multiple times independently."""

    def test_main_can_be_called_twice_with_different_tmp_paths(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Calling main() twice with separate tmp directories works correctly."""
        tmp_path_1 = tmp_path / "run1"
        tmp_path_1.mkdir()
        results_file_1 = tmp_path_1 / "tournament_report.json"
        data_dir_1 = tmp_path_1 / "data"
        q_tables_1 = {
            Team.ALPHA: data_dir_1 / "q_table_team_alpha.json",
            Team.BETA: data_dir_1 / "q_table_team_beta.json",
        }

        monkeypatch.setattr(
            "scripts.run_tournament.RESULTS_PATH", results_file_1
        )
        monkeypatch.setattr("scripts.run_tournament.DATA_DIR", data_dir_1)
        monkeypatch.setattr("scripts.run_tournament.Q_TABLE_PATHS", q_tables_1)

        main()

        assert results_file_1.exists()
        data_1 = json.loads(results_file_1.read_text())

        tmp_path_2 = tmp_path / "run2"
        tmp_path_2.mkdir()
        results_file_2 = tmp_path_2 / "tournament_report.json"
        data_dir_2 = tmp_path_2 / "data"
        q_tables_2 = {
            Team.ALPHA: data_dir_2 / "q_table_team_alpha.json",
            Team.BETA: data_dir_2 / "q_table_team_beta.json",
        }

        monkeypatch.setattr(
            "scripts.run_tournament.RESULTS_PATH", results_file_2
        )
        monkeypatch.setattr("scripts.run_tournament.DATA_DIR", data_dir_2)
        monkeypatch.setattr("scripts.run_tournament.Q_TABLE_PATHS", q_tables_2)

        main()

        assert results_file_2.exists()
        data_2 = json.loads(results_file_2.read_text())

        assert data_1["num_games"] == data_2["num_games"]
        assert len(data_1["games"]) == len(data_2["games"])
        assert set(data_1["final_scores"].keys()) == set(
            data_2["final_scores"].keys()
        )

    def test_main_successive_runs_produce_independent_q_tables(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Two successive main() calls produce distinct Q-table files."""
        tmp_path_1 = tmp_path / "run1"
        tmp_path_1.mkdir()
        results_file_1 = tmp_path_1 / "tournament_report.json"
        data_dir_1 = tmp_path_1 / "data"
        q_tables_1 = {
            Team.ALPHA: data_dir_1 / "q_table_team_alpha.json",
            Team.BETA: data_dir_1 / "q_table_team_beta.json",
        }

        monkeypatch.setattr(
            "scripts.run_tournament.RESULTS_PATH", results_file_1
        )
        monkeypatch.setattr("scripts.run_tournament.DATA_DIR", data_dir_1)
        monkeypatch.setattr("scripts.run_tournament.Q_TABLE_PATHS", q_tables_1)

        main()

        q_alpha_1 = json.loads(q_tables_1[Team.ALPHA].read_text())

        tmp_path_2 = tmp_path / "run2"
        tmp_path_2.mkdir()
        results_file_2 = tmp_path_2 / "tournament_report.json"
        data_dir_2 = tmp_path_2 / "data"
        q_tables_2 = {
            Team.ALPHA: data_dir_2 / "q_table_team_alpha.json",
            Team.BETA: data_dir_2 / "q_table_team_beta.json",
        }

        monkeypatch.setattr(
            "scripts.run_tournament.RESULTS_PATH", results_file_2
        )
        monkeypatch.setattr("scripts.run_tournament.DATA_DIR", data_dir_2)
        monkeypatch.setattr("scripts.run_tournament.Q_TABLE_PATHS", q_tables_2)

        main()

        q_alpha_2 = json.loads(q_tables_2[Team.ALPHA].read_text())

        assert isinstance(q_alpha_1, dict)
        assert isinstance(q_alpha_2, dict)
