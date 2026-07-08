"""Tool-schema logic for the Gridlock MCP server, decoupled from transport.

Kept independent of the `mcp` SDK (see server.py for the actual stdio
wiring) so this logic is directly testable — a test can construct a
`GameTools` around a fresh `MatchState` and call `get_observation`/
`make_move`/`get_match_status` as plain Python, without spinning up a real
MCP stdio client to exercise the same request/response flow described in
PRD §3.
"""

from __future__ import annotations

from engine.observation import Observation
from engine.player import Action, Role
from mcp_server.match_state import MatchState


def _serialize_observation(observation: Observation) -> dict:
    """Convert an `Observation` to a JSON-serializable dict.

    MCP tool responses are JSON over the wire, so the frozenset of barrier
    coordinates and the tuple-based positions are converted to lists here
    (once, at the serialization boundary) rather than leaking
    engine-internal types — frozensets and tuples-as-such don't round-trip
    through JSON — to every call site.
    """
    opponent = observation.opponent_position
    return {
        "self_position": list(observation.self_position),
        "opponent_position": list(opponent) if opponent is not None else None,
        "barriers": sorted(list(cell) for cell in observation.barriers),
        "move_count": observation.move_count,
    }


class GameTools:
    """Implements `get_observation`, `make_move`, `get_match_status`.

    Wraps a single `MatchState` so a server (or a test) can construct a
    fresh instance per match/test case instead of relying on shared
    mutable module-level state that would leak between episodes or tests.
    """

    def __init__(self, match: MatchState) -> None:
        self._match = match

    def get_observation(self, role: Role) -> dict:
        """Return `role`'s current fog-of-war observation."""
        return _serialize_observation(self._match.observation_for(role))

    def make_move(self, role: Role, direction: str) -> dict:
        """Queue `role`'s action for this turn; resolve once both are in.

        `direction` is validated against `Action`'s values here, before it
        ever reaches `GameLoop.step()` — `game_loop._resolve_move` only
        validates *board* legality (bounds/barriers) and has no defense
        against a malformed action shape, since it was designed to be
        called with a real `Action` already in hand. This is the last hop
        before that assumption would otherwise be violated by an external
        LLM caller's raw string.
        """
        try:
            action = Action(direction.strip().upper())
        except (AttributeError, ValueError):
            valid = ", ".join(member.value for member in Action)
            return {
                "error": f"invalid direction {direction!r}; expected one of {valid}"
            }

        try:
            result = self._match.submit_move(role, action)
        except RuntimeError as exc:
            return {"error": str(exc)}

        if result is None:
            return {"status": "waiting", "role": role.value, "action": action.value}

        observation = (
            result.cop_observation if role is Role.COP else result.thief_observation
        )
        reward = result.cop_reward if role is Role.COP else result.thief_reward
        return {
            "status": "resolved",
            "observation": _serialize_observation(observation),
            "move_count": result.move_count,
            "done": result.done,
            "reward": reward,
        }

    def get_match_status(self) -> dict:
        """Return move_count, done flag, and rewards if the episode ended."""
        return self._match.status()
