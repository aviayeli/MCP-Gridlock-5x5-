"""Load `game_config.json` into a typed, immutable configuration object.

Split out from game_loop.py purely to respect this package's 150-line-per
-file cap; the JSON-loading concern is also naturally separable from turn
-resolution logic, so nothing about game_loop.py's design depends on this
module living elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Resolves to <repo_root>/config/game_config.json regardless of cwd, since
# this file lives at <repo_root>/src/engine/config_loader.py.
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "game_config.json"
)


@dataclass(frozen=True)
class Scoring:
    """Terminal-reward payouts, taken verbatim from config (PRD §R)."""

    cop_win: int
    thief_win: int
    cop_loss: int
    thief_loss: int


@dataclass(frozen=True)
class GameConfig:
    """Typed view over `game_config.json`.

    A frozen dataclass is used instead of passing the raw dict around so
    every consumer gets attribute access and type checking rather than
    repeated stringly-typed lookups like `config["scoring"]["cop_win"]`,
    and so the config cannot be mutated mid-episode by accident.
    """

    rows: int
    cols: int
    max_moves: int
    num_games: int
    max_barriers: int
    scoring: Scoring


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> GameConfig:
    """Read and validate `game_config.json` from disk.

    Loaded at call time rather than cached at module-import time, so a
    caller (or a test) can point at an alternate config file by passing
    `path` instead of monkeypatching a module-level constant.
    """
    with Path(path).open() as config_file:
        raw = json.load(config_file)
    rows, cols = raw["grid_size"]
    return GameConfig(
        rows=rows,
        cols=cols,
        max_moves=raw["max_moves"],
        num_games=raw["num_games"],
        max_barriers=raw["max_barriers"],
        scoring=Scoring(**raw["scoring"]),
    )
