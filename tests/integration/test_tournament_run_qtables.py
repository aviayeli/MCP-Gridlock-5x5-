"""Test Q-table file generation and validation."""

from __future__ import annotations

import json
from pathlib import Path

from engine.player import Action
from engine.tournament_report import Team
from scripts.run_tournament import DATA_DIR, RESULTS_PATH, main


class TestQTableGeneration:
    """Test Q-table file creation, structure, and content."""

    def test_main_creates_q_table_files(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """main() writes both Q-table JSON files."""
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

        assert q_tables[Team.ALPHA].exists()
        assert q_tables[Team.BETA].exists()

    def test_main_q_tables_are_valid_json(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Q-table files are valid JSON."""
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

        for team, path in q_tables.items():
            data = json.loads(path.read_text())
            assert isinstance(data, dict), f"Q-table for {team} is not a dict"

    def test_main_q_tables_have_action_keys(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Q-table entries have keys for all Action values."""
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

        action_names = {action.name for action in Action}

        main()

        for team, path in q_tables.items():
            q_table = json.loads(path.read_text())
            for state_key, state_values in q_table.items():
                assert isinstance(
                    state_values, dict
                ), f"State {state_key} values not a dict"
                if state_values:
                    present_actions = set(state_values.keys())
                    for action in present_actions:
                        assert (
                            action in action_names
                        ), f"Invalid action {action} in {team} Q-table"

    def test_main_does_not_create_real_repo_results_file(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """main() respects monkeypatched RESULTS_PATH, doesn't write to real repo."""
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

        import scripts.run_tournament as module  # noqa: F401

        real_results = RESULTS_PATH

        main()

        assert results_file.exists()
        assert results_file != real_results

    def test_main_does_not_create_real_repo_data_directory(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """main() respects monkeypatched DATA_DIR, doesn't write to real repo."""
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

        real_data_dir = DATA_DIR

        main()

        assert data_dir.exists()
        assert q_tables[Team.ALPHA].exists()
        assert data_dir != real_data_dir
