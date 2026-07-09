"""Test game record fields, types, and role assignment."""

from __future__ import annotations

import json
from pathlib import Path

from engine.tournament_report import Outcome, Team
from scripts.run_tournament import main


class TestGameRecordFields:
    """Test that game records have correct fields and values."""

    def test_main_game_records_have_required_fields(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Each game record has game_index, teams, moves, rewards, outcome."""
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
        for game in data["games"]:
            assert "game_index" in game
            assert "cop_team" in game
            assert "thief_team" in game
            assert "moves" in game
            assert "cop_reward" in game
            assert "thief_reward" in game
            assert "outcome" in game

    def test_main_game_records_teams_are_strings(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Game cop_team and thief_team are serialized as strings."""
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
        for game in data["games"]:
            assert game["cop_team"] in ("ALPHA", "BETA")
            assert game["thief_team"] in ("ALPHA", "BETA")

    def test_main_game_records_outcome_is_valid_enum(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Each game's outcome is either CAPTURE or TIMEOUT."""
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
        valid_outcomes = {o.value for o in Outcome}
        for game in data["games"]:
            assert game["outcome"] in valid_outcomes

    def test_main_role_swap_schedule_holds(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Role-swap schedule alternates ALPHA/BETA as cop across games."""
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
        for i, game in enumerate(data["games"]):
            expected_cop = "ALPHA" if i % 2 == 0 else "BETA"
            expected_thief = "BETA" if i % 2 == 0 else "ALPHA"
            assert (
                game["cop_team"] == expected_cop
            ), f"Game {i}: cop_team mismatch"
            assert (
                game["thief_team"] == expected_thief
            ), f"Game {i}: thief_team mismatch"

    def test_main_game_records_opposite_roles(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """In each game, cop_team and thief_team are always opposite."""
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
        for game in data["games"]:
            cop = game["cop_team"]
            thief = game["thief_team"]
            assert cop != thief, "cop_team and thief_team should be opposite"
