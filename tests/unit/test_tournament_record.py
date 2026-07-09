"""Tests for GameRecord and build_game_record function."""

from __future__ import annotations

import pytest

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import TurnResult
from engine.observation import Observation
from engine.tournament_report import GameRecord, Outcome, Team, build_game_record


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
