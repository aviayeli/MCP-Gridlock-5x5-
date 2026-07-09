"""Tests for GameTools.make_move() validation, queuing, and resolution."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Role
from mcp_server.match_state import MatchState
from mcp_server.tools import GameTools


class TestGameToolsMakeMove:
    """Test GameTools.make_move() validation, queuing, and resolution."""

    def test_make_move_invalid_direction_returns_error_dict(self) -> None:
        """Invalid direction string returns error dict."""
        match = MatchState()
        tools = GameTools(match)

        result = tools.make_move(Role.COP, "FORWARD")
        assert isinstance(result, dict)
        assert "error" in result
        assert "FORWARD" in result["error"]

    def test_make_move_empty_string_returns_error_dict(self) -> None:
        """Empty direction string returns error dict."""
        match = MatchState()
        tools = GameTools(match)

        result = tools.make_move(Role.COP, "")
        assert isinstance(result, dict)
        assert "error" in result

    def test_make_move_returns_waiting_status_for_first_role(self) -> None:
        """First role's move returns waiting status."""
        match = MatchState()
        tools = GameTools(match)

        result = tools.make_move(Role.COP, "DOWN")
        assert result["status"] == "waiting"
        assert result["role"] == "COP"
        assert result["action"] == "DOWN"

    def test_make_move_returns_resolved_status_for_second_role(self) -> None:
        """Second role's move returns resolved status with observation."""
        match = MatchState()
        tools = GameTools(match)

        tools.make_move(Role.COP, "DOWN")
        result = tools.make_move(Role.THIEF, "STAY")

        assert result["status"] == "resolved"
        assert "observation" in result
        assert "move_count" in result
        assert "done" in result
        assert "reward" in result
        assert result["move_count"] == 1

    def test_make_move_case_insensitive(self) -> None:
        """Direction parsing is case-insensitive."""
        match = MatchState()
        tools = GameTools(match)

        result_lower = tools.make_move(Role.COP, "down")
        assert result_lower["status"] == "waiting"
        assert result_lower["action"] == "DOWN"

        # Reset for another test
        match = MatchState()
        tools = GameTools(match)
        result_mixed = tools.make_move(Role.COP, "DoWn")
        assert result_mixed["status"] == "waiting"
        assert result_mixed["action"] == "DOWN"

    def test_make_move_trims_whitespace(self) -> None:
        """Direction parsing trims leading/trailing whitespace."""
        match = MatchState()
        tools = GameTools(match)

        result = tools.make_move(Role.COP, "  UP  ")
        assert result["status"] == "waiting"
        assert result["action"] == "UP"

    def test_make_move_error_dict_when_match_already_done(self) -> None:
        """make_move returns error dict (not exception) after match is done."""
        config = GameConfig(
            rows=5, cols=5, max_moves=1, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 2))
        match = MatchState(game)
        tools = GameTools(match)

        # Play one turn to completion (and capture ends the game)
        tools.make_move(Role.COP, "STAY")
        tools.make_move(Role.THIEF, "STAY")

        # Attempt another move should return error dict
        result = tools.make_move(Role.COP, "STAY")
        assert isinstance(result, dict)
        assert "error" in result
        assert "cannot submit" in result["error"].lower()

    def test_make_move_includes_reward_field(self) -> None:
        """make_move includes reward field in resolved response."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        match = MatchState(game)
        tools = GameTools(match)

        tools.make_move(Role.COP, "STAY")
        result = tools.make_move(Role.THIEF, "STAY")

        # Result should include a reward field (None mid-game, int at end)
        assert "reward" in result
        assert result["reward"] is None or isinstance(result["reward"], int)

    def test_make_move_reward_is_integer_at_game_end(self) -> None:
        """make_move returns integer reward when game ends."""
        config = GameConfig(
            rows=5, cols=5, max_moves=2, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Place them far apart so they timeout at move 2
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        match = MatchState(game)
        tools = GameTools(match)

        # Turn 1: no game end yet
        tools.make_move(Role.COP, "STAY")
        result1 = tools.make_move(Role.THIEF, "STAY")
        assert result1["done"] is False
        assert result1["reward"] is None

        # Turn 2: game ends (max_moves=2)
        tools.make_move(Role.COP, "STAY")
        result2 = tools.make_move(Role.THIEF, "STAY")
        assert result2["done"] is True
        assert isinstance(result2["reward"], int)
        assert result2["reward"] in (20, 10, 5)  # Valid reward values
