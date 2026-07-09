"""Adapter reconciling `GameLoop`'s simultaneous `step()` with independent
MCP callers.

WHY this file exists: `GameLoop.step(cop_action, thief_action)` requires
both agents' actions at once, but the Cop and Thief are two independent
MCP clients that each call `make_move` on their own schedule, without ever
seeing the other's move first — that is the entire point of the Dec-POMDP
formulation this game implements (PRD §{Ai}, §O): neither agent may
condition its choice on the other's simultaneous choice. Buffering
whichever action arrives first, and only calling `step()` once both are
in, makes the *server* — not either agent — the synchronization point,
matching the "shared game state store" in the network architecture
(PRD §3). This is the one piece of adaptation the MCP integration needs
on top of the engine as written.
"""

from __future__ import annotations

from engine.game_loop import GameLoop, TurnResult
from engine.observation import Observation, build_observation
from engine.player import Action, Role


class MatchState:
    """Owns one live `GameLoop` plus the pending-action buffer for one turn.

    A fresh `MatchState` is created per match (and per test) rather than
    kept as module-level global state, so concurrent or successive
    episodes — and unit tests run against this class — never share
    mutable state by accident.
    """

    def __init__(self, game_loop: GameLoop | None = None) -> None:
        """Set up the match state with a pending-action buffer."""
        self._game = game_loop if game_loop is not None else GameLoop()
        self._pending: dict[Role, Action] = {}
        self._last_result: TurnResult | None = None

    @property
    def game(self) -> GameLoop:
        """Expose the underlying `GameLoop` read-only, for status queries."""
        return self._game

    def observation_for(self, role: Role) -> Observation:
        """Return `role`'s fog-of-war view of the *current* game state.

        Recomputed from live board/player state on every call (rather than
        cached from the last resolved turn) so `get_observation` reflects
        reality even before the first turn of the match has resolved, when
        no `TurnResult` yet exists.
        """
        cop_pos = self._game.cop.position
        thief_pos = self._game.thief.position
        barriers = self._game.board.barrier_coordinates()
        move_count = self._game.move_count
        if role is Role.COP:
            return build_observation(cop_pos, thief_pos, barriers, move_count)
        return build_observation(thief_pos, cop_pos, barriers, move_count)

    def submit_move(self, role: Role, action: Action) -> TurnResult | None:
        """Record `role`'s action; resolve the turn once both roles are in.

        Returns `None` while the turn is still waiting on the other role's
        action (the "queued" case the MCP tool must report back to its
        caller), or the resolved `TurnResult` once `GameLoop.step()` has
        run for this turn. A role may overwrite its own pending action by
        calling again before its partner arrives — that's a revision, not
        an error, since no commitment is observable until both are in.
        """
        if self._game.done:
            raise RuntimeError("cannot submit a move: the match has already ended")

        self._pending[role] = action
        if len(self._pending) < len(Role):
            return None

        result = self._game.step(
            self._pending.pop(Role.COP), self._pending.pop(Role.THIEF)
        )
        self._last_result = result
        return result

    def status(self) -> dict:
        """Return move_count/done, plus rewards once the episode has ended.

        Rewards stay `None` until `_last_result` exists and the episode is
        actually over, mirroring `TurnResult`'s own `None`-until-terminal
        contract rather than inventing a different sentinel here.
        """
        cop_reward = thief_reward = None
        if self._last_result is not None:
            cop_reward = self._last_result.cop_reward
            thief_reward = self._last_result.thief_reward
        return {
            "move_count": self._game.move_count,
            "done": self._game.done,
            "cop_reward": cop_reward,
            "thief_reward": thief_reward,
        }
