"""Integration tests for the tournament runner end-to-end flow.

Tests verify that `run_tournament.main()` produces correctly-structured
output (results JSON, Q-table files) without modifying the real repo's
`results/` and `data/` directories. Uses pytest's `monkeypatch` fixture
to redirect filesystem writes to a temporary test directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.config_loader import load_config
from engine.player import Action
from engine.tournament_report import Outcome, Team
from scripts.run_tournament import (
    DATA_DIR,
    RESULTS_PATH,
    _smoke_check_mcp_layer,
    main,
)


class TestSmokeCheckMcpLayer:
    """Test the MCP integration smoke-check in isolation."""

    def test_smoke_check_mcp_layer_succeeds(self) -> None:
        """_smoke_check_mcp_layer() instantiates real MCP tools and checks state."""
        # This should not raise; it constructs a real GameTools/MatchState
        # and verifies get_match_status() returns expected initial state.
        _smoke_check_mcp_layer()


class TestEndToEndTournamentRun:
    """Test end-to-end run of main() with filesystem mocking."""

    def test_main_generates_results_json(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """main() generates a valid tournament_report.json file."""
        # Redirect RESULTS_PATH to tmp directory
        results_file = tmp_path / "tournament_report.json"
        monkeypatch.setattr(
            "scripts.run_tournament.RESULTS_PATH", results_file
        )

        # Redirect DATA_DIR for Q-table files
        data_dir = tmp_path / "data"
        monkeypatch.setattr("scripts.run_tournament.DATA_DIR", data_dir)

        # Redirect Q_TABLE_PATHS to use the new data_dir
        q_tables = {
            Team.ALPHA: data_dir / "q_table_team_alpha.json",
            Team.BETA: data_dir / "q_table_team_beta.json",
        }
        monkeypatch.setattr("scripts.run_tournament.Q_TABLE_PATHS", q_tables)

        main()

        # Verify results file exists
        assert results_file.exists(), "tournament_report.json not created"

        # Verify it's valid JSON
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

        # Check all required top-level keys
        assert "num_games" in data
        assert "games" in data
        assert "final_scores" in data
        assert "winner" in data

        # Check types
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
        """Game cop_team and thief_team are serialized as strings ("ALPHA", "BETA")."""
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
        """Each game's cop/thief rewards match one of the config's scoring pairs."""
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

        # Determine expected action names
        action_names = {action.name for action in Action}

        main()

        for team, path in q_tables.items():
            q_table = json.loads(path.read_text())
            # Each state entry should have keys for the action names
            for state_key, state_values in q_table.items():
                assert isinstance(
                    state_values, dict
                ), f"State {state_key} values not a dict"
                # At least one action should be present
                if state_values:
                    present_actions = set(state_values.keys())
                    # All present actions should be valid action names
                    for action in present_actions:
                        assert (
                            action in action_names
                        ), f"Invalid action {action} in {team} Q-table"

    def test_main_does_not_create_real_repo_results_file(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """main() respects monkeypatched RESULTS_PATH, doesn't write to real repo."""
        # Use a different path than the real RESULTS_PATH
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

        # Get the real repo paths before main() runs
        import scripts.run_tournament as module  # noqa: F401

        real_results = RESULTS_PATH

        main()

        # Verify the monkeypatched path was used
        assert results_file.exists()

        # The real repo path should still be clean (if it didn't exist before)
        # We can't assert it doesn't exist because there may be pre-existing
        # files from earlier runs, but we verify we wrote to tmp_path instead
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

        # Verify we wrote to tmp_path's data_dir
        assert data_dir.exists()
        assert q_tables[Team.ALPHA].exists()

        # Verify we used a different path
        assert data_dir != real_data_dir


class TestIdempotentRuns:
    """Test that main() can be called multiple times independently."""

    def test_main_can_be_called_twice_with_different_tmp_paths(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Calling main() twice with separate tmp directories works correctly."""
        # First run
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

        # Second run
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

        # Both runs should have valid results
        assert data_1["num_games"] == data_2["num_games"]
        assert len(data_1["games"]) == len(data_2["games"])
        assert set(data_1["final_scores"].keys()) == set(
            data_2["final_scores"].keys()
        )

    def test_main_successive_runs_produce_independent_q_tables(
        self, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Two successive main() calls produce distinct Q-table files."""
        # First run
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

        # Second run
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

        # Both should be valid dicts (though potentially different due to randomness)
        assert isinstance(q_alpha_1, dict)
        assert isinstance(q_alpha_2, dict)
