"""JSON (de)serialization for `QLearningAgent.q_table` (PRD §{Ωi}).

Split out from q_agent.py purely to respect this package's 150-line-per-file
cap; "how do we turn a Q-table into JSON and back" is also a self-contained
concern, separate from the learning/policy logic that owns `q_table` itself.

WHY this string encoding for `StateKey`
----------------------------------------
JSON object keys must be strings, but `StateKey` (see state_encoding.py) is a
`tuple[RelativeOpponent, int]` where `RelativeOpponent` is itself either a
`(int, int)` tuple or `None`. Rather than invent a bespoke delimiter format
(e.g. `f"{dr},{dc},{mask}"`, which needs its own escaping rules for the
`None` case), this module keys the JSON object on `json.dumps(state_key)`
directly: `json.dumps((None, 9))` -> `"[null, 9]"`,
`json.dumps(((1, 0), 9))` -> `"[[1, 0], 9]"`. This is deterministic (the same
`StateKey` always dumps to the same string), unambiguous (no two distinct
`StateKey` values can collide, since `json.dumps` output is a faithful
structural encoding), and trivially round-trippable (`json.loads` back into
a list, then a small conversion step turns the outer/inner lists back into
tuples and `None` back into `None` -- see `_decode_state_key`). It also
reuses the standard library's own tested encoder instead of hand-rolling
string parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from engine.player import Action

if TYPE_CHECKING:
    from agents.state_encoding import RelativeOpponent, StateKey

# Intended caller convention (not hardcoded/enforced here -- see q_agent.py
# and this module's docstring): callers persist one table per role at
# `data/q_table_cop.json` and `data/q_table_thief.json`.


def _encode_state_key(state_key: StateKey) -> str:
    """Serialize one `StateKey` to its JSON-object-key string form."""
    return json.dumps(state_key)


def _decode_state_key(key_str: str) -> StateKey:
    """Inverse of `_encode_state_key`: JSON string -> exact original `StateKey`."""
    opponent_raw, barrier_mask = json.loads(key_str)
    opponent_relative: RelativeOpponent = (
        None if opponent_raw is None else (opponent_raw[0], opponent_raw[1])
    )
    return (opponent_relative, barrier_mask)


def _encode_values(values: dict[Action, float]) -> dict[str, float]:
    """Inner `dict[Action, float]` -> `{action.value: q_value}` (JSON-safe)."""
    return {action.value: value for action, value in values.items()}


def _decode_values(values: dict[str, float]) -> dict[Action, float]:
    """Inverse of `_encode_values`."""
    return {Action(action_str): value for action_str, value in values.items()}


def save_q_table(
    path: Path | str, q_table: dict[StateKey, dict[Action, float]]
) -> None:
    """Write `q_table` to `path` as JSON, creating parent directories as needed.

    Each `StateKey` becomes a JSON object key via `_encode_state_key`; each
    inner `dict[Action, float]` becomes a plain `{action.value: q_value}`
    object, since `Action` (an `Enum`) is not natively JSON-serializable.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        _encode_state_key(state_key): _encode_values(values)
        for state_key, values in q_table.items()
    }
    target.write_text(json.dumps(serializable, indent=2))


def load_q_table(path: Path | str) -> dict[StateKey, dict[Action, float]]:
    """Read a Q-table JSON file back into `dict[StateKey, dict[Action, float]]`.

    Missing file = cold start: if `path` does not exist, this returns `{}`
    instead of raising, so a fresh agent (or an agent whose table hasn't been
    saved yet this run) simply starts with no prior learning rather than
    crashing the caller.
    """
    target = Path(path)
    if not target.exists():
        return {}

    raw = json.loads(target.read_text())
    return {
        _decode_state_key(key_str): _decode_values(values)
        for key_str, values in raw.items()
    }
