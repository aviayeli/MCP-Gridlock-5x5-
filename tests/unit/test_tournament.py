"""Tests for Tournament system: play scheduling, scoring, and report generation."""

from __future__ import annotations

import json

import pytest

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import TurnResult
from engine.observation import Observation
from engine.player import Action
from engine.tournament import Tournament
from engine.tournament_report import (
    GameRecord,
    Outcome,
    Team,
    build_game_record,
    build_report,
)


class TestTournamentInit:
    """Test Tournament.__init__ validation."""

    def test_init_raises_valueerror_missing_team(self) -> None:
        """Tournament raises ValueError if policies missing one team."""
        policies = {Team.ALPHA: lambda obs: Action.STAY}
        with pytest.raises(ValueError, match="exactly one entry per Team"):
            Tournament(policies)

    def test_init_raises_valueerror_extra_team(self) -> None:
        """Tournament raises ValueError if policies have extra entries."""
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
            "GAMMA": lambda obs: Action.STAY,  # type: ignore
        }
        with pytest.raises(ValueError, match="exactly one entry per Team"):
            Tournament(policies)

    def test_init_raises_valueerror_empty_policies(self) -> None:
        """Tournament raises ValueError if policies is empty."""
        with pytest.raises(ValueError, match="exactly one entry per Team"):
            Tournament({})

    def test_init_succeeds_with_exactly_two_teams(self) -> None:
        """Tournament initializes successfully with one policy per Team."""
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies)
        assert tournament.policies == policies

    def test_init_default_config(self) -> None:
        """Tournament uses default config if not provided."""
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies)
        assert tournament.config is not None
        assert tournament.config.num_games == 6

    def test_init_custom_config(self) -> None:
        """Tournament accepts custom config."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=4, max_barriers=5,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        assert tournament.config.num_games == 4

    def test_init_custom_start_positions(self) -> None:
        """Tournament accepts custom start positions."""
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(
            policies, cop_start=(1, 1), thief_start=(3, 3)
        )
        assert tournament.cop_start == (1, 1)
        assert tournament.thief_start == (3, 3)

    def test_init_initializes_empty_records_and_zero_scores(self) -> None:
        """Tournament starts with empty records and zero scores."""
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies)
        assert tournament.records == []
        assert tournament.team_scores == {Team.ALPHA: 0, Team.BETA: 0}


class TestTournamentRoleSwapSchedule:
    """Test that Tournament alternates Cop/Thief assignment across games."""

    def test_role_swap_schedule_alpha_starts_cop(self) -> None:
        """Team ALPHA plays Cop on even-indexed games."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=6, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Use always-STAY policy to ensure predictable results
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        # Check cop_team assignment: ALPHA on games 0, 2, 4; BETA on 1, 3, 5
        assert tournament.records[0].cop_team == Team.ALPHA
        assert tournament.records[1].cop_team == Team.BETA
        assert tournament.records[2].cop_team == Team.ALPHA
        assert tournament.records[3].cop_team == Team.BETA
        assert tournament.records[4].cop_team == Team.ALPHA
        assert tournament.records[5].cop_team == Team.BETA

    def test_thief_team_is_opposite_of_cop_team(self) -> None:
        """Thief team is always opposite of Cop team."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=6, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        for record in tournament.records:
            if record.cop_team == Team.ALPHA:
                assert record.thief_team == Team.BETA
            else:
                assert record.thief_team == Team.ALPHA


class TestTournamentScoreAccumulation:
    """Test that scores accumulate correctly across games."""

    def test_always_stay_produces_timeout_in_every_game(self) -> None:
        """With always-STAY policies, both players stay put and timeout."""
        config = GameConfig(
            rows=5, cols=5, max_moves=2, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        # With one game and always-STAY starting from (0,0) and (4,4):
        # They never meet, so timeout occurs
        assert len(tournament.records) == 1
        record = tournament.records[0]
        assert record.outcome == Outcome.TIMEOUT
        # Timeout: Cop gets cop_loss (5), Thief gets thief_win (10)
        assert record.cop_reward == 5
        assert record.thief_reward == 10

    def test_score_accumulation_balanced_series(self) -> None:
        """With always-STAY and 6 games, scores should be equal (tie)."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=6, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        # Each team plays 3 games as Cop and 3 as Thief
        # As Cop (timeout): cop_loss=5
        # As Thief (timeout): thief_win=10
        # Total per team: 3*5 + 3*10 = 15 + 30 = 45
        assert tournament.team_scores[Team.ALPHA] == 45
        assert tournament.team_scores[Team.BETA] == 45

    def test_report_final_scores_match_team_scores(self) -> None:
        """Report final_scores match team_scores."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=2, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        report = tournament.report()
        assert report["final_scores"]["ALPHA"] == tournament.team_scores[Team.ALPHA]
        assert report["final_scores"]["BETA"] == tournament.team_scores[Team.BETA]


class TestTournamentWinner:
    """Test winner determination in reports."""

    def test_winner_tie_with_balanced_scores(self) -> None:
        """Report shows 'tie' when final scores are equal."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=6, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        report = tournament.report()
        assert report["winner"] == "tie"

    def test_winner_determined_by_higher_score(self) -> None:
        """Report shows winner with higher cumulative score."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )

        def alpha_policy(obs: Observation) -> Action:  # noqa: ARG001
            """ALPHA always captures (gets cop_win=20 as Cop)."""
            return Action.DOWN

        def beta_policy(obs: Observation) -> Action:  # noqa: ARG001
            """BETA always stays (gets thief_loss or thief_win)."""
            return Action.STAY

        policies = {Team.ALPHA: alpha_policy, Team.BETA: beta_policy}
        tournament = Tournament(policies, config=config)
        tournament.play()

        report = tournament.report()
        # ALPHA plays Cop in game 0 and should score higher
        if report["final_scores"]["ALPHA"] > report["final_scores"]["BETA"]:
            assert report["winner"] == "ALPHA"
        elif report["final_scores"]["BETA"] > report["final_scores"]["ALPHA"]:
            assert report["winner"] == "BETA"


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


class TestGameRecord:
    """Test GameRecord creation and serialization."""

    def test_game_record_to_dict_serializes_enums(self) -> None:
        """GameRecord.to_dict() renders Enums as .value strings."""
        record = GameRecord(
            game_index=0,
            cop_team=Team.ALPHA,
            thief_team=Team.BETA,
            moves=10,
            cop_reward=20,
            thief_reward=5,
            outcome=Outcome.CAPTURE,
        )
        record_dict = record.to_dict()

        assert record_dict["cop_team"] == "ALPHA"
        assert record_dict["thief_team"] == "BETA"
        assert record_dict["outcome"] == "CAPTURE"

    def test_game_record_to_dict_has_all_fields(self) -> None:
        """GameRecord.to_dict() includes all expected fields."""
        record = GameRecord(
            game_index=2,
            cop_team=Team.BETA,
            thief_team=Team.ALPHA,
            moves=15,
            cop_reward=5,
            thief_reward=10,
            outcome=Outcome.TIMEOUT,
        )
        record_dict = record.to_dict()

        assert record_dict["game_index"] == 2
        assert record_dict["moves"] == 15
        assert record_dict["cop_reward"] == 5
        assert record_dict["thief_reward"] == 10

    def test_game_record_frozen_prevents_mutation(self) -> None:
        """GameRecord is frozen and cannot be mutated."""
        record = GameRecord(
            game_index=0,
            cop_team=Team.ALPHA,
            thief_team=Team.BETA,
            moves=10,
            cop_reward=20,
            thief_reward=5,
            outcome=Outcome.CAPTURE,
        )
        with pytest.raises(AttributeError):
            record.game_index = 1  # type: ignore


class TestBuildGameRecord:
    """Test build_game_record function."""

    def test_build_game_record_infers_capture_outcome(self) -> None:
        """build_game_record infers CAPTURE when cop_reward == cop_win."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        result = TurnResult(
            cop_observation=Observation((0, 0), (0, 0), frozenset(), 1),
            thief_observation=Observation((0, 0), (0, 0), frozenset(), 1),
            move_count=1,
            done=True,
            cop_reward=20,  # cop_win
            thief_reward=5,
        )
        record = build_game_record(0, Team.ALPHA, Team.BETA, result, config)

        assert record.outcome == Outcome.CAPTURE

    def test_build_game_record_infers_timeout_outcome(self) -> None:
        """build_game_record infers TIMEOUT when cop_reward == cop_loss."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        result = TurnResult(
            cop_observation=Observation((0, 0), None, frozenset(), 25),
            thief_observation=Observation((4, 4), None, frozenset(), 25),
            move_count=25,
            done=True,
            cop_reward=5,  # cop_loss
            thief_reward=10,
        )
        record = build_game_record(0, Team.ALPHA, Team.BETA, result, config)

        assert record.outcome == Outcome.TIMEOUT

    def test_build_game_record_copies_scores(self) -> None:
        """build_game_record copies cop/thief rewards exactly."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        result = TurnResult(
            cop_observation=Observation((0, 0), (0, 0), frozenset(), 1),
            thief_observation=Observation((0, 0), (0, 0), frozenset(), 1),
            move_count=1,
            done=True,
            cop_reward=20,
            thief_reward=5,
        )
        record = build_game_record(3, Team.BETA, Team.ALPHA, result, config)

        assert record.cop_reward == 20
        assert record.thief_reward == 5


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


class TestTournamentToJson:
    """Test Tournament.to_json() produces valid JSON."""

    def test_to_json_produces_valid_json_string(self) -> None:
        """to_json() returns a valid JSON string."""
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

        json_str = tournament.to_json()
        assert isinstance(json_str, str)

        # Should be parseable JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_to_json_matches_report_contents(self) -> None:
        """to_json() output parses back to equivalent report() dict."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=2, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        policies = {
            Team.ALPHA: lambda obs: Action.STAY,
            Team.BETA: lambda obs: Action.STAY,
        }
        tournament = Tournament(policies, config=config)
        tournament.play()

        report = tournament.report()
        json_str = tournament.to_json()
        parsed = json.loads(json_str)

        # The parsed JSON should match report() output
        assert parsed == report

    def test_to_json_is_indented(self) -> None:
        """to_json() produces indented, human-readable output."""
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

        json_str = tournament.to_json()
        # Indented JSON should contain newlines and spaces
        assert "\n" in json_str
        assert "  " in json_str


class TestTournamentPlayMethod:
    """Test Tournament.play() orchestration."""

    def test_play_creates_records_for_all_games(self) -> None:
        """play() creates one GameRecord per num_games."""
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

        assert len(tournament.records) == 3

    def test_play_updates_team_scores(self) -> None:
        """play() accumulates scores into team_scores."""
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

        # After play(), team_scores should be non-zero
        assert tournament.team_scores[Team.ALPHA] > 0
        assert tournament.team_scores[Team.BETA] > 0

    def test_play_sets_game_indices_in_order(self) -> None:
        """Games in records are indexed 0, 1, 2, ... in play order."""
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

        for i, record in enumerate(tournament.records):
            assert record.game_index == i
