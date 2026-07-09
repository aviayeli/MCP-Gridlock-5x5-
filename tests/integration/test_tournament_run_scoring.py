"""Test final scores, winner determination, and reward consistency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.config_loader import load_config
from engine.tournament_report import Outcome, Team
from scripts.run_tournament import main


class TestScoringAndRewards:
    """Test final scores, winner, and reward pair consistency."""

    def test_main_final_scores_keys_are_team_names(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """final_scores dict has exactly 'ALPHA' and 'BETA' keys."""
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
        assert set(data["final_scores"].keys()) == {"ALPHA", "BETA"}

    def test_main_final_scores_are_integers(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """final_scores values are integers."""
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
        for team, score in data["final_scores"].items():
            assert isinstance(score, int), f"Score for {team} is not int"

    def test_main_winner_is_valid_value(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """winner is one of 'ALPHA', 'BETA', or 'tie'."""
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
        assert data["winner"] in ("ALPHA", "BETA", "tie")

    def test_main_reward_pairs_are_consistent_with_config(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Each game's cop/thief rewards match one of config's scoring pairs."""
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
        for game in data["games"]:
            cop_reward = game["cop_reward"]
            thief_reward = game["thief_reward"]
            outcome = game["outcome"]

            if outcome == Outcome.CAPTURE.value:
                assert cop_reward == config.scoring.cop_win
                assert thief_reward == config.scoring.thief_loss
            elif outcome == Outcome.TIMEOUT.value:
                assert cop_reward == config.scoring.cop_loss
                assert thief_reward == config.scoring.thief_win
            else:
                pytest.fail(f"Unknown outcome: {outcome}")
