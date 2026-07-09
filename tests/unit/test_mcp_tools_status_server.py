"""Tests for GameTools.get_match_status() and server.py module-level tools."""

from __future__ import annotations

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Role
from mcp_server.match_state import MatchState
from mcp_server.server import get_match_status, get_observation, make_move
from mcp_server.tools import GameTools


class TestGameToolsGetMatchStatus:
    """Test GameTools.get_match_status() returns correct dict."""

    def test_get_match_status_shape_before_game_end(self) -> None:
        """get_match_status returns correct shape with None rewards."""
        match = MatchState()
        tools = GameTools(match)

        status = tools.get_match_status()
        assert isinstance(status, dict)
        assert "move_count" in status
        assert "done" in status
        assert "cop_reward" in status
        assert "thief_reward" in status
        assert status["cop_reward"] is None
        assert status["thief_reward"] is None

    def test_get_match_status_shape_after_game_end(self) -> None:
        """get_match_status returns rewards after game ends."""
        config = GameConfig(
            rows=5, cols=5, max_moves=1, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 2))
        match = MatchState(game)
        tools = GameTools(match)

        tools.make_move(Role.COP, "STAY")
        tools.make_move(Role.THIEF, "STAY")

        status = tools.get_match_status()
        assert status["done"] is True
        assert status["cop_reward"] == 20
        assert status["thief_reward"] == 5


class TestServerModuleLevelTools:
    """Test server.py's module-level tool functions."""

    def test_get_observation_valid_role_cop(self) -> None:
        """get_observation works with 'COP' role."""
        # Note: server.py has module-level state; this test works with the shared state
        result = get_observation("COP")
        assert isinstance(result, dict)
        assert "self_position" in result
        assert "opponent_position" in result
        assert "barriers" in result
        assert "move_count" in result

    def test_get_observation_valid_role_thief(self) -> None:
        """get_observation works with 'THIEF' role."""
        result = get_observation("THIEF")
        assert isinstance(result, dict)
        assert "self_position" in result

    def test_get_observation_invalid_role_returns_error(self) -> None:
        """get_observation returns error dict for invalid role."""
        result = get_observation("WIZARD")
        assert isinstance(result, dict)
        assert "error" in result
        assert "WIZARD" in result["error"]

    def test_get_observation_case_insensitive(self) -> None:
        """get_observation parses role case-insensitively."""
        result_lower = get_observation("cop")
        assert "error" not in result_lower
        assert "self_position" in result_lower

    def test_get_observation_trims_whitespace(self) -> None:
        """get_observation trims whitespace from role string."""
        result = get_observation("  COP  ")
        assert "error" not in result
        assert "self_position" in result

    def test_make_move_valid_role(self) -> None:
        """make_move works with valid role."""
        result = make_move("COP", "STAY")
        # Could be waiting or resolved depending on order, but no error
        assert "error" not in result

    def test_make_move_invalid_role_returns_error(self) -> None:
        """make_move returns error dict for invalid role."""
        result = make_move("INVALID", "UP")
        assert isinstance(result, dict)
        assert "error" in result

    def test_make_move_case_insensitive_role(self) -> None:
        """make_move parses role case-insensitively."""
        result = make_move("thief", "STAY")
        assert "error" not in result

    def test_get_match_status_returns_dict(self) -> None:
        """get_match_status returns a valid dict."""
        result = get_match_status()
        assert isinstance(result, dict)
        assert "move_count" in result
        assert "done" in result
        assert "cop_reward" in result
        assert "thief_reward" in result

    def test_server_tools_are_callable_directly(self) -> None:
        """Verify that server.py's tool functions are callable without MCP transport."""
        # Plain Python functions, decorated with @mcp.tool() but still callable
        result = get_observation("COP")
        assert isinstance(result, dict)
