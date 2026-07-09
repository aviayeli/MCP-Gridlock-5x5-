"""Smoke test for MCP layer integration."""

from __future__ import annotations

from scripts.run_tournament import _smoke_check_mcp_layer


class TestSmokeCheckMcpLayer:
    """Test the MCP integration smoke-check in isolation."""

    def test_smoke_check_mcp_layer_succeeds(self) -> None:
        """_smoke_check_mcp_layer() instantiates real MCP tools and checks state."""
        # This should not raise; it constructs a real GameTools/MatchState
        # and verifies get_match_status() returns expected initial state.
        _smoke_check_mcp_layer()
