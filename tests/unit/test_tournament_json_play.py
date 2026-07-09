"""Tests for Tournament.to_json() and play() method."""

from __future__ import annotations

import json

from engine.config_loader import GameConfig, Scoring
from engine.player import Action
from engine.tournament import Tournament
from engine.tournament_report import Team


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
