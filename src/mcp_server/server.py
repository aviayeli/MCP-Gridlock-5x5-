"""stdio MCP server exposing the Gridlock match to independent LLM agents.

Uses `mcp.server.fastmcp.FastMCP`, the SDK's high-level decorator-based API
for defining tools, rather than the lower-level `mcp.server.Server`
primitives — this server needs nothing beyond "register a few functions as
tools and serve them over stdio" (PRD §3's request/response flow), which is
exactly FastMCP's intended use case; the lower-level API only earns its
extra ceremony when a server needs custom lifecycle/session control this
one does not.

WHY role validation happens here, at the transport boundary, rather than
inside tools.py: `role` arrives from an external LLM caller as a raw JSON
string over stdio, before any Python type system can catch a malformed
value. tools.py's `GameTools` methods are typed to accept a real `Role`
enum member precisely so that once execution is inside tools.py, "role" is
an established fact rather than a string every method must re-validate —
converting/rejecting it once, right where untrusted input enters the
system, is what preserves that guarantee for the rest of the codebase.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from engine.player import Role
from mcp_server.match_state import MatchState
from mcp_server.tools import GameTools

mcp = FastMCP("gridlock-5x5")
_match = MatchState()
_tools = GameTools(_match)


def _parse_role(role: str) -> Role | None:
    """Convert a raw role string to `Role`, or `None` if it isn't one."""
    try:
        return Role(role.strip().upper())
    except (AttributeError, ValueError):
        return None


@mcp.tool()
def get_observation(role: str) -> dict:
    """Return the calling agent's current fog-of-war observation.

    `role` must be "COP" or "THIEF".
    """
    parsed = _parse_role(role)
    if parsed is None:
        return {"error": f"invalid role {role!r}; expected COP or THIEF"}
    return _tools.get_observation(parsed)


@mcp.tool()
def make_move(role: str, direction: str) -> dict:
    """Submit `role`'s action for the current turn.

    Resolves the turn once both Cop and Thief have submitted; until then,
    returns a "waiting" status rather than the resolved outcome.
    """
    parsed = _parse_role(role)
    if parsed is None:
        return {"error": f"invalid role {role!r}; expected COP or THIEF"}
    return _tools.make_move(parsed, direction)


@mcp.tool()
def get_match_status() -> dict:
    """Return current move_count, done flag, and rewards if terminal."""
    return _tools.get_match_status()


def main() -> None:
    """Run the MCP server over stdio (the transport PRD §3 assumes)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
