# PRD: MCP-Gridlock-5x5 — Dual AI Agent Race via MCP

## 1. Overview / Motivation

MCP-Gridlock-5x5 is an experiment in **dual-agent coordination and competition
via the Model Context Protocol (MCP)**. Two small local LLMs — one playing a
Cop, one playing a Thief — chase each other around a 5x5 grid dotted with
barriers. Each agent's move is decided by a locally-hosted inference engine
(e.g. Ollama) and submitted to a cloud-exposed MCP server that owns the
authoritative game state.

The central research question is not "can an LLM play a game well" but
**"can a small local LLM reliably drive a deterministic strategy game by
issuing structured tool calls, rather than free-form natural-language
reasoning?"** The game itself (Cops and Thieves on a grid) is intentionally
simple and fully specified so that the interesting variable is the
agent/tool-calling loop, not the game design. A Q-Table based reinforcement
learning layer sits underneath each agent's policy so that performance can
improve across the `num_games` (6) games of a run and be measured
objectively (win rate, moves-to-capture, token spend per decision).

This project doubles as a FinOps case study: proving that a full multi-agent,
multi-server, multi-game experiment can be run entirely on small local models
and structured tool schemas, with cloud resources used only for MCP transport
(ngrok tunnels), not for inference.

## 2. Formal Dec-POMDP Definition

The game is modeled as a **Decentralized Partially Observable Markov Decision
Process (Dec-POMDP)**, formally the tuple:

```
〈 n, S, {Ai}, P, R, {Ωi}, O, γ 〉
```

### n — Number of agents
`n = 2`. Agent 1 is the **Cop**, Agent 2 is the **Thief**. Each is driven by
its own local LLM policy and talks to its own dedicated MCP server (Cop
server / Thief server, see §3). This is a general-sum, competitive Dec-POMDP:
the Cop and Thief have opposing objectives, not a shared reward.

### S — State space
A state `s ∈ S` fully describes the world at a given instant:

- `cop_pos ∈ {0,...,4} × {0,...,4}` — Cop's grid coordinate on the 5x5 board
  (`grid_size: [5, 5]` in `config/game_config.json`).
- `thief_pos ∈ {0,...,4} × {0,...,4}` — Thief's grid coordinate.
- `barriers ⊆ grid cells`, `|barriers| ≤ max_barriers` (5) — fixed,
  impassable cells placed at episode start and held constant for that
  episode.
- `move_count ∈ {0, ..., max_moves}` — number of turns elapsed this episode
  (`max_moves: 25`).

`S` is therefore finite: `25 (cop positions) × 25 (thief positions) ×
C(25, ≤5) (barrier layouts) × 26 (move counts)`, which is small enough for an
exact tabular Q-representation (no function approximation needed).

### {Ai} — Per-agent action sets
Each agent has an identical action set:

```
A_cop = A_thief = { UP, DOWN, LEFT, RIGHT, STAY }
```

A move is legal only if the destination cell is within the 5x5 bounds and is
not an occupied barrier cell; illegal moves are rejected by the MCP server
and re-prompted (or resolved as STAY, per the server's validation contract —
see PLAN.md for the exact resolution rule chosen). `STAY` is always legal and
gives the RL agent a genuine "wait" option, which matters for luring/ambush
strategies.

### P — Transition function
`P(s' | s, a_cop, a_thief)` is **deterministic**, not stochastic: given the
current state and both agents' chosen (validated) actions, the next
positions are computed by simple grid arithmetic (clamped to bounds, blocked
by barriers) and `move_count` increments by 1. Determinism is a deliberate
design choice — see PLAN.md §"Determinism & Reproducibility" — so that
identical Q-tables and identical LLM outputs reproduce identical games,
which is required for grading and debugging.

### R — Reward function
Reward is terminal-sparse, taken directly from the `scoring` object in
`config/game_config.json`:

| Outcome | Cop reward | Thief reward |
|---|---|---|
| Cop captures Thief (same cell) | `cop_win = 20` | `thief_loss = 5` |
| `max_moves` reached without capture | `cop_loss = 5` | `thief_win = 10` |

This is intentionally **asymmetric**: a Cop win (20) is worth twice as much
to the Cop as a Thief win (10) is to the Thief, while both "loss" payouts are
equal (5). The design implication is that the Cop is incentivized to hunt
aggressively (the high upside for capture dominates its policy), while the
Thief's incentive structure rewards survival more modestly but still
positively — the Thief is never punished to zero, which avoids a
degenerate "give up" policy during Q-learning. The equal loss values (5/5)
mean neither agent is catastrophically penalized for losing, which keeps
early, near-random Q-tables from collapsing into a single dominant strategy
before enough episodes have been played to differentiate good policies from
bad ones.

### {Ωi} — Per-agent observation spaces
This is what makes the process a **Dec-POMDP rather than a Dec-MDP**. The
design choice made here is:

- **The Cop observes**: its own position, the full barrier layout, and the
  Thief's position **only within a fixed Manhattan-distance visibility
  radius** (fog of war outside that radius — the Cop does not see the exact
  cell, only "not currently visible").
- **The Thief observes**: its own position, the full barrier layout, and the
  Cop's position under the same visibility-radius rule.
- Both agents always observe `move_count` (turns remaining), since that is
  a shared, freely-visible game-clock signal, not tactical information.

This partial-observability rule is deliberately symmetric (same radius for
both roles) so neither side has an inherent information advantage — the
asymmetry in the game lives entirely in the reward function (§R), not in
sensing. Full barrier visibility is justified because barriers are static
and set once per episode (treating them as "public knowledge terrain"
keeps the state space tractable and keeps the interesting uncertainty
confined to the opponent's position, which is the crux of a cops-and-thieves
game).

### O — Observation function
`O(ωi | s', a_cop, a_thief)` maps the post-transition state and the joint
action to each agent's observation. Given the deterministic visibility-radius
rule above, `O` is itself deterministic (not noisy): if the opponent's
Manhattan distance from agent `i` is within the visibility radius, `ωi`
includes the opponent's exact cell; otherwise `ωi` reports the opponent as
"unseen" for that turn. This keeps `O` simple and fully specified while still
producing genuine partial observability at the state-space level, since each
agent's Q-table is indexed by its own observation, not the true global
state.

### γ — Discount factor
`γ = 0.9`. A discount below 1 is used because rewards are terminal and
episodes are capped at `max_moves` (25) turns; `γ = 0.9` gives the Cop's
Q-learning a meaningful incentive to capture sooner rather than later
(discounted future reward shrinks each turn), which produces a more
decisive, less passive Cop policy than `γ = 1` would.

## 3. Network Architecture

```
 ┌─────────────────────┐                       ┌─────────────────────┐
 │   Local machine A    │                       │   Local machine B    │
 │  Ollama (small model) │                       │  Ollama (small model) │
 │   Cop policy + Q-table│                       │ Thief policy + Q-table│
 └──────────┬────────────┘                       └───────────┬───────────┘
            │ MCP tool calls                                  │ MCP tool calls
            ▼                                                 ▼
 ┌─────────────────────┐                       ┌─────────────────────┐
 │   Cop MCP Server      │◄──── shared game ────►│  Thief MCP Server    │
 │ (cloud-exposed via     │       state store      │ (cloud-exposed via   │
 │  ngrok / equivalent)   │                       │  ngrok / equivalent) │
 └─────────────────────┘                       └─────────────────────┘
```

Each agent's local inference engine (Ollama or an equivalent small-model
runtime) hosts the policy that decides the agent's next action; the two MCP
servers are the authoritative source of truth for game state and are reached
over cloud tunnels (ngrok or equivalent), so the two agents can run on
different machines/networks while playing the same game.

**Single-turn request/response flow:**

1. **Query state** — the local LLM (via its agent runtime) issues an MCP
   tool call, e.g. `get_state()`, to its own server (Cop → Cop server,
   Thief → Thief server). The server responds with that agent's
   partially-observed view `ωi` (own position, barrier layout, opponent
   position if within visibility radius, move count).
2. **Decide** — the local LLM, prompted with a compact structured
   representation of `ωi` and the legal action set `Ai`, selects one action.
   The decision is informed by the agent's Q-table (looked up by observation
   key) rather than by open-ended reasoning.
3. **Submit action** — the agent issues a second MCP tool call, e.g.
   `submit_action(action="UP")`, back to its server.
4. **Validate & update** — the server validates the action against the
   current state (bounds, barriers), applies the deterministic transition
   `P`, updates the shared game state, checks terminal conditions (capture,
   or `move_count == max_moves`), and returns the turn result (new
   observation, reward if terminal, done flag) to the calling agent.

Both agents repeat this loop until the episode ends; `num_games` (6) such
episodes make up one full experiment run.

## 4. FinOps Emphasis

Strict token/cost management is a first-class requirement, not an
afterthought:

- **Small local models only for the per-turn decision loop.** All in-game
  action selection runs on a small model served locally via Ollama (or
  equivalent) — no per-turn calls to metered cloud LLM APIs. Cloud
  infrastructure (ngrok tunnels) is used purely for MCP transport, never for
  inference billing.
- **Structured tool-call schemas, not free-form chat.** Both the
  observation payload (`get_state` response) and the action payload
  (`submit_action` request) are fixed, minimal JSON schemas — grid
  coordinates, an enum of 5 actions, barrier list, move count — rather than
  natural-language descriptions the model must parse and generate. This
  keeps both prompt and completion tokens per turn small and bounded,
  independent of how verbose a general-purpose chat model might otherwise
  be.
- **Config-bounded token spend.** `max_moves` (25) and `num_games` (6) from
  `config/game_config.json` give a hard ceiling on the number of decision
  turns per run: at most `25 moves × 2 agents × 6 games = 300` LLM
  invocations per experiment, each against a small, structured,
  low-token-count prompt. This bound is what makes total per-run token (and
  therefore cost/energy) spend predictable and auditable ahead of time,
  which is the point of the FinOps framing — cost is a designed-in property
  of the config, not an emergent surprise.

## 5. Success Criteria

- Both Cop and Thief agents complete all `num_games` (6) games per run
  without protocol errors (malformed tool calls, schema violations, timeouts).
- Q-tables show measurable learning across games within a run (e.g. Cop's
  average moves-to-capture decreases, or Cop win rate increases, from game 1
  to game 6).
- End-to-end turn latency and token count per decision stay within a
  small-model budget (no fallback to a larger/cloud model required to keep
  the game moving).
- Full episode logs are reproducible: given the same starting Q-tables,
  barrier layout, and LLM sampling seed, replaying a game yields the same
  sequence of moves (supports grading/debugging).

## 6. Out of Scope

- Training or fine-tuning the underlying LLM weights — the local model is
  used as-is; only the Q-table (not the model) is learned/updated.
- Grid sizes, barrier counts, or scoring values other than those defined in
  `config/game_config.json`; those are configuration, not hard-coded product
  requirements, but this PRD does not design for arbitrary grid sizes.
- More than 2 concurrent agents/roles (no third "witness" or multi-cop
  variants in this phase).
- A graphical/interactive UI; this experiment is evaluated via logs and
  summary metrics, not a rendered game client.
- Any production deployment concerns (auth, multi-tenant server hosting,
  rate limiting) beyond what's needed to run the experiment end-to-end.
