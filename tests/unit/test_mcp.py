"""Tests for MCP server subsystem: MatchState, GameTools, and server.py tools."""

from __future__ import annotations

import pytest

from engine.config_loader import GameConfig, Scoring
from engine.game_loop import GameLoop
from engine.player import Action, Role
from mcp_server.match_state import MatchState
from mcp_server.server import get_match_status, get_observation, make_move
from mcp_server.tools import GameTools


class TestMatchStateObservation:
    """Test MatchState.observation_for() returns correct fog-of-war view."""

    def test_observation_before_any_move(self) -> None:
        """Observation reflects initial positions even before first turn."""
        match = MatchState()
        cop_obs = match.observation_for(Role.COP)
        thief_obs = match.observation_for(Role.THIEF)

        # Cop sees itself at (0,0), Thief at (4,4) out of visibility range
        assert cop_obs.self_position == (0, 0)
        # Distance (0,0) to (4,4) is 8, exceeds VISIBILITY_RADIUS=2
        assert cop_obs.opponent_position is None

        # Thief sees itself at (4,4), Cop out of visibility range
        assert thief_obs.self_position == (4, 4)
        assert thief_obs.opponent_position is None

        # Both see barriers (empty frozenset or populated depending on board)
        assert isinstance(cop_obs.barriers, frozenset)
        assert isinstance(thief_obs.barriers, frozenset)

        # Move count is 0
        assert cop_obs.move_count == 0
        assert thief_obs.move_count == 0

    def test_observation_fog_of_war_out_of_range(self) -> None:
        """Opponent position is None when outside visibility radius."""
        # Create custom config with max_barriers=0 to make placement deterministic
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Cop at (0,0), Thief at (4,4) are far apart (manhattan distance = 8)
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        match = MatchState(game)

        cop_obs = match.observation_for(Role.COP)
        thief_obs = match.observation_for(Role.THIEF)

        # Distance is 8, visibility radius is 2, so opponent is unseen
        assert cop_obs.opponent_position is None
        assert thief_obs.opponent_position is None

    def test_observation_fog_of_war_in_range(self) -> None:
        """Opponent position is visible when within visibility radius."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Cop at (2,2), Thief at (2,4) are 2 apart (manhattan distance = 2)
        # This is exactly at VISIBILITY_RADIUS
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 4))
        match = MatchState(game)

        cop_obs = match.observation_for(Role.COP)
        thief_obs = match.observation_for(Role.THIEF)

        # Both should see the opponent
        assert cop_obs.opponent_position == (2, 4)
        assert thief_obs.opponent_position == (2, 2)

    def test_observation_updates_after_move(self) -> None:
        """Observation reflects updated positions after a move."""
        match = MatchState()
        # First move: Cop moves UP (to -1, but clamped to 0), Thief stays
        # Actually UP from (0,0) is invalid, so Cop stays at (0,0)
        result = match.submit_move(Role.COP, Action.UP)
        assert result is None  # Waiting for Thief
        result = match.submit_move(Role.THIEF, Action.STAY)
        assert result is not None  # Both submitted, resolved

        # After first turn, move_count should be 1
        cop_obs = match.observation_for(Role.COP)
        assert cop_obs.move_count == 1


class TestMatchStateSubmitMove:
    """Test MatchState.submit_move() buffering and resolution."""

    def test_submit_move_returns_none_waiting_for_other_role(self) -> None:
        """submit_move returns None while waiting for the other role."""
        match = MatchState()
        result = match.submit_move(Role.COP, Action.DOWN)
        assert result is None

    def test_submit_move_returns_turn_result_when_both_submitted(self) -> None:
        """submit_move returns TurnResult once both roles have submitted."""
        match = MatchState()
        result_cop = match.submit_move(Role.COP, Action.DOWN)
        assert result_cop is None

        result_thief = match.submit_move(Role.THIEF, Action.RIGHT)
        assert result_thief is not None
        # Verify it's a TurnResult-like object with expected fields
        assert hasattr(result_thief, "move_count")
        assert hasattr(result_thief, "done")
        assert hasattr(result_thief, "cop_observation")
        assert hasattr(result_thief, "thief_observation")
        assert result_thief.move_count == 1

    def test_buffer_clears_for_next_turn(self) -> None:
        """Buffer clears after resolving a turn, allowing turn 2."""
        match = MatchState()

        # Turn 1
        result1 = match.submit_move(Role.COP, Action.DOWN)
        assert result1 is None
        result1 = match.submit_move(Role.THIEF, Action.STAY)
        assert result1 is not None
        assert result1.move_count == 1

        # Turn 2: should resolve as a fresh turn, not stale state
        result2 = match.submit_move(Role.COP, Action.STAY)
        assert result2 is None  # Waiting for Thief again
        result2 = match.submit_move(Role.THIEF, Action.STAY)
        assert result2 is not None
        assert result2.move_count == 2  # Confirm move_count incremented

    def test_role_can_revise_pending_action_before_partner_arrives(self) -> None:
        """A role can overwrite its pending action before the turn resolves."""
        match = MatchState()
        result1 = match.submit_move(Role.COP, Action.DOWN)
        assert result1 is None

        # Cop revises its move from DOWN to RIGHT
        result_revised = match.submit_move(Role.COP, Action.RIGHT)
        assert result_revised is None  # Still waiting

        # Thief submits; the revised Cop action (RIGHT) should be used
        result = match.submit_move(Role.THIEF, Action.STAY)
        assert result is not None
        # Cop starts at (0,0), moves RIGHT -> (0,1)
        assert result.cop_observation.self_position == (0, 1)

    def test_submit_move_raises_runtime_error_after_done(self) -> None:
        """submit_move raises RuntimeError after the match is done."""
        config = GameConfig(
            rows=5, cols=5, max_moves=1, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Start close together to guarantee capture on first turn
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 2))
        match = MatchState(game)

        # First move: they're at same position, so game ends
        result = match.submit_move(Role.COP, Action.STAY)
        assert result is None
        result = match.submit_move(Role.THIEF, Action.STAY)
        assert result is not None
        assert result.done is True

        # Attempt to submit a move after done should raise RuntimeError
        with pytest.raises(RuntimeError, match="cannot submit a move"):
            match.submit_move(Role.COP, Action.STAY)


class TestMatchStateStatus:
    """Test MatchState.status() returns correct game state."""

    def test_status_before_any_move(self) -> None:
        """status() returns correct initial state."""
        match = MatchState()
        status = match.status()

        assert status["move_count"] == 0
        assert status["done"] is False
        assert status["cop_reward"] is None
        assert status["thief_reward"] is None

    def test_status_mid_game(self) -> None:
        """status() returns move_count but None rewards mid-game."""
        match = MatchState()
        match.submit_move(Role.COP, Action.STAY)
        match.submit_move(Role.THIEF, Action.STAY)

        status = match.status()
        assert status["move_count"] == 1
        assert status["done"] is False
        assert status["cop_reward"] is None
        assert status["thief_reward"] is None

    def test_status_after_game_end(self) -> None:
        """status() returns rewards after game ends."""
        config = GameConfig(
            rows=5, cols=5, max_moves=1, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 2))
        match = MatchState(game)

        match.submit_move(Role.COP, Action.STAY)
        match.submit_move(Role.THIEF, Action.STAY)

        status = match.status()
        assert status["done"] is True
        assert status["cop_reward"] == 20  # cop_win for capture
        assert status["thief_reward"] == 5  # thief_loss for capture


class TestGameToolsGetObservation:
    """Test GameTools.get_observation() returns correct dict shape."""

    def test_get_observation_returns_dict_with_correct_shape(self) -> None:
        """get_observation returns a dict with expected keys."""
        match = MatchState()
        tools = GameTools(match)

        cop_obs = tools.get_observation(Role.COP)
        assert isinstance(cop_obs, dict)
        assert "self_position" in cop_obs
        assert "opponent_position" in cop_obs
        assert "barriers" in cop_obs
        assert "move_count" in cop_obs

    def test_get_observation_serializes_positions_as_lists(self) -> None:
        """Positions are lists, not tuples (or None if out of range)."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        # Place them 2 apart (exactly at VISIBILITY_RADIUS)
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 4))
        tools = GameTools(MatchState(game))

        cop_obs = tools.get_observation(Role.COP)
        assert isinstance(cop_obs["self_position"], list)
        assert cop_obs["self_position"] == [2, 2]
        assert isinstance(cop_obs["opponent_position"], list)
        assert cop_obs["opponent_position"] == [2, 4]

    def test_get_observation_serializes_barriers_as_list_of_lists(self) -> None:
        """Barriers are a list of [row, col] lists."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=5,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config)
        tools = GameTools(MatchState(game))

        obs = tools.get_observation(Role.COP)
        assert isinstance(obs["barriers"], list)
        # Each barrier should be a list of 2 integers
        for barrier in obs["barriers"]:
            assert isinstance(barrier, list)
            assert len(barrier) == 2
            assert isinstance(barrier[0], int)
            assert isinstance(barrier[1], int)

    def test_get_observation_fog_of_war(self) -> None:
        """get_observation respects fog-of-war (opponent_position None when unseen)."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(0, 0), thief_start=(4, 4))
        tools = GameTools(MatchState(game))

        cop_obs = tools.get_observation(Role.COP)
        # Distance 8 > VISIBILITY_RADIUS, so opponent_position should be None
        assert cop_obs["opponent_position"] is None

    def test_get_observation_fog_of_war_in_range(self) -> None:
        """get_observation reveals opponent when in range."""
        config = GameConfig(
            rows=5, cols=5, max_moves=25, num_games=1, max_barriers=0,
            scoring=Scoring(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5),
        )
        game = GameLoop(config=config, cop_start=(2, 2), thief_start=(2, 4))
        tools = GameTools(MatchState(game))

        cop_obs = tools.get_observation(Role.COP)
        # Distance 2 == VISIBILITY_RADIUS, should see opponent
        assert cop_obs["opponent_position"] == [2, 4]


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
