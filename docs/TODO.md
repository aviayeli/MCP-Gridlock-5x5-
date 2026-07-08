# TODO: MCP-Gridlock-5x5 Roadmap

Roadmap for building the Dual AI Agent Race via MCP experiment, per
`docs/PRD.md` and `docs/PLAN.md`. Organized into 5 phases.

## Phase 1: Setup

- [ ] Confirm `pyproject.toml` dependencies cover: MCP server/client library,
      Ollama client (or HTTP client for its API), numpy (if used for
      Q-tables), pytest, ruff.
- [ ] Verify `.env.example` documents all required environment variables
      (Ollama host/port, model name, ngrok auth token, MCP server ports).
- [ ] Confirm `config/game_config.json` is loaded by a single shared config
      module rather than re-parsed/hardcoded in multiple places.
- [ ] Scaffold `src/` package layout: `game/`, `mcp_servers/`, `agents/`,
      `rl/`, `cli/`.
- [ ] Scaffold `tests/` package layout mirroring `src/`.
- [ ] Set up local Ollama instance with a small model pulled and confirmed
      reachable (`ollama list` / a smoke-test completion call).
- [ ] Verify `results/` and `docs/` are correctly git-tracked (not ignored)
      so experiment outputs and planning docs persist.

## Phase 2: Local Game Engine (Grid & Rules)

- [ ] Implement grid state representation (`grid_size`, cop position, thief
      position, barrier set, move count) sourced from
      `config/game_config.json`.
- [ ] Implement seeded barrier placement (up to `max_barriers` = 5),
      excluding start cells.
- [ ] Implement seeded starting-position assignment for Cop and Thief.
- [ ] Implement the action set `{UP, DOWN, LEFT, RIGHT, STAY}` and legality
      checking (grid bounds, barrier occupancy).
- [ ] Implement the deterministic transition function `P` (apply joint
      action → next state).
- [ ] Implement terminal-condition checks: capture (cop_pos == thief_pos)
      and timeout (`move_count == max_moves` = 25).
- [ ] Implement reward resolution using the `scoring` object (`cop_win`=20,
      `thief_win`=10, `cop_loss`=5, `thief_loss`=5).
- [ ] Implement per-agent observation derivation (visibility-radius fog of
      war per PRD §{Ωi}).
- [ ] Unit tests: bounds checking, barrier blocking, capture detection,
      timeout detection, reward correctness, observation/fog-of-war
      correctness.

## Phase 3: MCP Server Integration

- [ ] Define MCP tool schema for `get_state()` (structured observation
      response).
- [ ] Define MCP tool schema for `submit_action(action)` (structured
      request/response, validation result).
- [ ] Implement Cop MCP server wrapping the shared grid engine.
- [ ] Implement Thief MCP server wrapping the shared grid engine.
- [ ] Implement simultaneous-turn synchronization between the two servers
      (hold-and-resolve-together, not first-caller-wins).
- [ ] Add server-side validation rejecting illegal actions with a clear
      error/response rather than corrupting state.
- [ ] Smoke-test both servers locally with a scripted/random client (no LLM
      yet) to confirm protocol correctness end-to-end.
- [ ] Expose each MCP server via ngrok tunnel (or equivalent) and confirm
      remote reachability.
- [ ] Integration tests covering a full scripted episode over the MCP
      protocol (not just in-process engine calls).

## Phase 4: RL Agent Strategy (Q-Learning)

- [ ] Implement Q-table data structure (dict keyed by observation tuple →
      5-length Q-value array).
- [ ] Implement ε-greedy action selection over the Q-table, with the local
      LLM consuming the structured observation to inform/select the action.
- [ ] Implement the Q-learning update rule (`α`, `γ` = 0.9 per PRD, using
      rewards from `config/game_config.json`).
- [ ] Implement ε decay schedule across the `num_games` (6) games of a run.
- [ ] Implement Q-table persistence to/from `results/` (save after each
      game; support resuming a run).
- [ ] Wire the local LLM (Ollama) into the agent decision loop using the
      structured tool-call schemas (no free-form prompts) from Phase 3.
- [ ] Pin LLM sampling (temperature 0 / fixed seed) for reproducibility.
- [ ] Unit tests: Q-table update math, ε decay correctness, persistence
      round-trip (save → load → identical table).

## Phase 5: Multiplayer Testing & Logs

- [ ] Implement the experiment runner (`cli/`) that drives all `num_games`
      (6) games end-to-end against both MCP servers.
- [ ] Implement per-turn logging (state, observation, action, validation
      result) and per-game logging (outcome, reward, moves-to-capture).
- [ ] Persist per-episode seeds (barrier layout, start positions) alongside
      logs for reproducibility.
- [ ] Run a full multi-game experiment against locally-hosted MCP servers
      and confirm no protocol errors across all 6 games.
- [ ] Re-run the same experiment against ngrok-tunneled (cloud-exposed) MCP
      servers and confirm parity with the local run.
- [ ] Verify reproducibility: replaying a run with the same seeds and
      starting Q-tables yields an identical move sequence.
- [ ] Verify learning trend: Cop win rate / moves-to-capture improves across
      games 1 through 6 within a run.
- [ ] Measure and record token usage per turn/per run against the FinOps
      budget implied by `max_moves` (25) × `num_games` (6).
- [ ] Write up final experiment results summary in `results/`.
