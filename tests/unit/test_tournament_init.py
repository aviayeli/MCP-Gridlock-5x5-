"""Tests for Tournament initialization and role-swap schedule."""

from __future__ import annotations

import pytest

from engine.config_loader import GameConfig, Scoring
from engine.player import Action
from engine.tournament import Tournament
from engine.tournament_report import Team


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
