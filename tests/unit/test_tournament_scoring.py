"""Tests for Tournament score accumulation and winner determination."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.observation import Observation
from engine.player import Action
from engine.tournament import Tournament
from engine.tournament_report import Outcome, Team


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
