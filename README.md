# MCP-Gridlock-5x5

**Dual AI Agent Race via MCP** — a Decentralized POMDP (Dec-POMDP) game of Cops and Thieves on a 5x5 grid, where two independently-learning Q-learning agents compete over a Model Context Protocol (MCP) tool interface, swapping roles across a multi-game tournament.

## 1. Overview

Two agents — **Team Alpha** and **Team Beta** — play a 6-game series of Cops and Thieves on a 5x5 grid dotted with barriers. Each game, one team plays Cop (trying to capture the Thief) and the other plays Thief (trying to survive `max_moves` turns); the roles swap every game, so across a series both teams experience both roles equally. Each team is driven by its own tabular Q-learning agent, which observes a partial, fog-of-war view of the board and chooses a move via an MCP-shaped tool call (`get_observation` / `make_move` / `get_match_status`).

The project is a small, fully-specified, deterministic testbed for a broader question: can independent agents, observing only a partial view of a shared environment and acting through a narrow, structured tool interface, learn a competitive policy from sparse terminal rewards alone? The game is intentionally simple so that the interesting variable is the agent/tool-calling loop, not the game design — see `docs/PRD.md` for the full motivation and FinOps framing (small models, structured schemas, and bounded token spend, rather than expensive free-form reasoning, are the design's central constraint).

## 2. Formal Dec-POMDP Definition

The game is modeled as a Decentralized Partially Observable Markov Decision Process:

```
⟨ n, S, {Ai}, P, R, {Ωi}, O, γ ⟩
```

All values below are the actual defaults shipped in this repo (`config/game_config.json`, `src/agents/state_encoding.py`, `src/agents/q_agent.py`) — not illustrative placeholders.

### n — Number of agents
`n = 2`: one Cop, one Thief. Which *team* (Alpha/Beta) holds which role alternates every game (Alpha plays Cop on even game indices, Beta on odd), so over a 6-game series each team plays 3 games as Cop and 3 as Thief.

### S — State space
A state fully describes: the Cop's grid coordinate, the Thief's grid coordinate (each in a 5×5 grid, from `grid_size` in `config/game_config.json`), the fixed barrier layout (up to `max_barriers = 5` cells, placed once per episode and held constant), and the elapsed move count (0 to `max_moves = 25`). This is finite and small enough for an exact tabular representation — no function approximation is used anywhere in this codebase.

### {Ai} — Action set
Both roles share the same 5-action set: `{UP, DOWN, LEFT, RIGHT, STAY}` (`src/engine/player.py`, `Action` enum). `STAY` is a first-class action (not merely "no move"), since it supports genuine wait/ambush/lure tactics.

### P — Transition function
Deterministic: given both agents' chosen actions, the next state is computed by simple grid arithmetic. An illegal move (out of the 5×5 bounds, or onto a barrier cell) does **not** raise or get rejected mid-turn — it resolves to `STAY` (the player's position doesn't change that turn). This keeps `P` total (every state/action pair yields a next state) and deterministic, which is what makes replaying an episode with the same starting conditions reproducible (`src/engine/game_loop.py`, `GameLoop._resolve_move`).

### R — Reward function
Reward is **sparse and terminal-only** — `None` on every non-terminal turn, populated only when an episode ends. Taken directly from `config/game_config.json`'s `scoring` object:

| Outcome | Cop reward | Thief reward |
|---|---|---|
| Capture (Cop and Thief occupy the same cell) | `cop_win = 20` | `thief_loss = 5` |
| Timeout (`max_moves` reached without capture) | `cop_loss = 5` | `thief_win = 10` |

The reward is intentionally asymmetric: a Cop win (20) is worth twice a Thief win (10), incentivizing the Cop to hunt aggressively, while both "loss" payouts are equal (5) so neither role is punished to zero for losing — this avoids a degenerate "give up" policy while Q-tables are still under-trained.

### {Ωi} / O — Partial observability (fog of war)
This is what makes the process a Dec-POMDP rather than a Dec-MDP. Each agent's observation always includes its own exact position, the full static barrier layout (barriers are fixed per episode, so treating them as public knowledge keeps the interesting uncertainty confined to the opponent's position), and the current move count. The opponent's position is revealed **only if within a fixed Manhattan-distance visibility radius**, `VISIBILITY_RADIUS = 2` (`src/engine/observation.py`) — otherwise the observation reports the opponent as unseen (`None`). The visibility rule is symmetric (same radius for both roles), so the game's only built-in asymmetry is in the reward function, not in sensing.

### γ — Discount factor
`γ = 0.9` (`QLearningAgent`'s default in `src/agents/q_agent.py`), giving the Q-learning update a meaningful incentive to resolve episodes sooner rather than later.

### From Dec-POMDP state to Q-table key
A raw observation (absolute self/opponent positions plus the full barrier list) is far larger than necessary for a usable Q-table and doesn't generalize across the board. `src/agents/state_encoding.py` compresses each observation into a small, hashable key:

- The opponent's position is encoded as a **relative delta** `(Δrow, Δcol)` from the agent's own position, or `None` if unseen (fog-of-war) — this collapses geometrically-identical situations at different absolute board positions onto the same key, so a tactic learned in one corner of the grid transfers immediately to an equivalent situation elsewhere.
- Barriers are encoded as a **4-bit adjacency mask** (is a barrier immediately UP/DOWN/LEFT/RIGHT of the agent right now) rather than full relative barrier positions — only an *immediately adjacent* barrier can affect this turn's legal moves, so this is a deliberate, lossy simplification that keeps the state space tiny (14 possible opponent-delta values × 16 possible barrier masks = 224 keys) instead of combinatorially large.
- `move_count` is excluded from the key entirely — it's a monotonic game clock, not tactical information a one-step Q-lookup can act on differently.

## 3. Architecture

The system is built in three layers, each with its own test suite:

### 3.1 Game engine (`src/engine/`)
Strict OOP, single-responsibility modules, each ≤150 lines:
- `board.py` — `Board`: bounds/barrier checks, seeded random barrier placement excluding given start cells.
- `player.py` — `Player`, `Action`, `Role`: position tracking and pure move arithmetic (no legality checking — that's the caller's job).
- `config_loader.py` — loads `config/game_config.json` into a typed, immutable `GameConfig`.
- `observation.py` — builds each agent's fog-of-war `Observation` from the live board/player state.
- `game_loop.py` — `GameLoop`: turn resolution, terminal-condition checks, reward assignment for a single episode.
- `tournament.py` / `tournament_report.py` — `Tournament`: plays a full `num_games`-game series, alternates which team plays which role every game, accumulates per-team scores, and assembles the final JSON report.

### 3.2 MCP server (`src/mcp_server/`)
A stdio [`FastMCP`](https://github.com/modelcontextprotocol) server (`server.py`) exposing three tools to any connecting MCP client: `get_observation(role)`, `make_move(role, direction)`, `get_match_status()`. Because the Cop and Thief are two *independent* MCP clients that each call `make_move` on their own schedule — without ever seeing the other's move first, which is the whole point of the Dec-POMDP framing — `match_state.py`'s `MatchState` class buffers whichever action arrives first and only resolves the turn (calling `GameLoop.step()`) once both roles have submitted. This makes the server itself, not either agent, the synchronization point for simultaneous turns.

### 3.3 Q-learning agents (`src/agents/`)
- `state_encoding.py` — the Dec-POMDP-to-Q-table-key compression described above.
- `q_agent.py` — `QLearningAgent`: epsilon-greedy action selection (with injectable `random.Random` for reproducible exploration) and an explicit, caller-triggered `decay_epsilon()`.
- `q_update.py` — the exact Bellman/TD-error update: `Q[s,a] += α · (reward + γ·max(Q[next]) − Q[s,a])` for non-terminal transitions, and `Q[s,a] += α · (reward − Q[s,a])` for terminal ones (the discounted future term is dropped entirely on a terminal transition — a classic, easy-to-get-backwards detail in tabular Q-learning).
- `q_persistence.py` — JSON save/load for a Q-table, round-tripping the composite `(relative_opponent, barrier_mask)` state keys exactly.

### 3.4 How the pieces connect (and one deliberate simplification)
`src/scripts/run_tournament.py`, the tournament runner, drives `Tournament` **directly against `GameLoop`** — it does *not* route each turn through the live MCP stdio transport. This is a deliberate choice, not an oversight: the MCP server's `MatchState` buffering exists to reconcile two *independent, asynchronous* external clients, which is a different concurrency model than an offline, same-process batch trainer needs. The runner does still instantiate the real `GameTools`/`MatchState` objects as a smoke check that the MCP integration is correctly wired to the same underlying engine, but game-driving itself goes through `Tournament`'s already-tested synchronous simultaneous-turn resolution. The stdio MCP server (`src/mcp_server/server.py`) remains a separate, fully working, interactive entry point for real external LLM/agent clients — running the tournament script does not start or require it.

Because `Tournament.policies` is keyed by **team**, not role (teams swap Cop/Thief every game), each team gets exactly one `QLearningAgent` — not one per role. Since the state encoding above is already role-agnostic (a relative delta and a barrier mask carry no built-in notion of "I am the Cop"), a single per-team Q-table legitimately accumulates experience from both roles that team takes on across the series. Q-tables are persisted to `data/q_table_team_alpha.json` and `data/q_table_team_beta.json`. Because rewards are sparse and terminal-only, each team performs exactly one Bellman update per game — using its last chosen `(state, action)` and that game's actual terminal reward — followed by one `decay_epsilon()` call, once the game ends.

## 4. Usage

Install dependencies (this is a `uv`-managed project; `[tool.uv] package = false` since it's an application, not a distributed library — `uv sync` sets up the virtual environment and the `pytest`/`ruff` dev dependency group):

```bash
uv sync
```

Run the tournament. The project has no installed console-script entry point, and `pyproject.toml`'s `pythonpath = ["src"]` config only applies inside `pytest` — a plain `uv run python` invocation needs `PYTHONPATH` set explicitly to resolve the `engine`/`mcp_server`/`agents`/`scripts` package imports:

```bash
PYTHONPATH=src uv run python src/scripts/run_tournament.py
```

This plays the full `num_games`-game series (6, per `config/game_config.json`), writes the series report to `results/tournament_report.json`, and saves each team's learned Q-table to `data/q_table_team_alpha.json` / `data/q_table_team_beta.json`.

Run the test suite and lint:

```bash
uv run pytest --cov=src --cov-report=term-missing
uv run ruff check .
```

## 5. Tournament Results

The committed `results/tournament_report.json` reflects the most recent run:

| Game | Cop | Thief | Moves | Cop reward | Thief reward | Outcome |
|---|---|---|---|---|---|---|
| 0 | ALPHA | BETA | 25 | 5 | 10 | TIMEOUT |
| 1 | BETA | ALPHA | 25 | 5 | 10 | TIMEOUT |
| 2 | ALPHA | BETA | 25 | 5 | 10 | TIMEOUT |
| 3 | BETA | ALPHA | 25 | 5 | 10 | TIMEOUT |
| 4 | ALPHA | BETA | 25 | 5 | 10 | TIMEOUT |
| 5 | BETA | ALPHA | 25 | 5 | 10 | TIMEOUT |

**Final scores:** Team Alpha 45, Team Beta 45 — **winner: tie**.

The role-swap schedule alternates correctly (Alpha/Beta/Alpha/Beta/Alpha/Beta), and the perfectly symmetric 45/45 result is consistent with the schedule's fairness-by-construction, not a coincidence.

Honest caveat: `QLearningAgent` defaults to `epsilon = 1.0` (full exploration) decaying by a factor of `0.9` per game down to a floor of `0.05`, and `run_tournament.py` does not currently pin a random seed. Over only 6 games, epsilon barely leaves the high-exploration regime (≈0.9 → ≈0.59 by the last decay), so this committed result reflects **near-random play, not a converged learned policy** — every game timed out with no capture. The underlying Bellman-update math, state encoding, and persistence are all independently unit-tested and verified correct (see `tests/unit/test_q_agent_*.py`); a longer series, a lower/faster-decaying starting epsilon, or a fixed seed would be needed to demonstrate a clear learning trend, and `docs/TODO.md` notes this honestly as a partially-verified item rather than claiming more than the current run shows.

## 6. Project layout

```
config/game_config.json   # grid size, move/game limits, barrier count, scoring — no hardcoded values elsewhere
src/engine/                # Board, Player, GameLoop, Tournament — the OOP game engine
src/mcp_server/             # FastMCP stdio server + MatchState turn-buffering
src/agents/                 # Q-learning agent: state encoding, epsilon-greedy, Bellman update, persistence
src/scripts/                # run_tournament.py — the offline tournament/training driver
tests/unit/, tests/integration/   # 228 tests, 99% coverage on src/
docs/PRD.md, PLAN.md, TODO.md     # design rationale and phase-by-phase build log
results/, data/              # tournament_report.json and learned Q-tables (generated by running the script)
```

Every `.py` file in this repo — source and test — is kept at or under 150 lines, favoring many small, single-responsibility modules over large ones.
