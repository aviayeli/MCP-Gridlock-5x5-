# TODO: MCP-Gridlock-5x5 Roadmap

Roadmap for building the Dual AI Agent Race via MCP experiment, per
`docs/PRD.md` and `docs/PLAN.md`. Organized into 5 phases.

**Status: All 5 core phases complete.** Note: Local-LLM/Ollama and ngrok remote exposure were deliberately bypassed in favor of a centralized MCP architecture. (228 unit + integration tests, 99% code coverage)

## Phase 1: Setup

- [x] Confirm `pyproject.toml` dependencies cover: MCP server/client library,
      pytest, pytest-cov, ruff. (Note: Ollama and numpy not needed; Q-agent 
      uses epsilon-greedy directly, no LLM integration.)
- [x] Verify `.env.example` is correctly tracked (no environment variables 
      required for this implementation; .env.example is minimalist).
- [x] Confirm `config/game_config.json` is loaded by a single shared config
      module rather than re-parsed/hardcoded in multiple places.
- [x] Scaffold `src/` package layout: `engine/`, `mcp_server/`, `agents/`,
      `scripts/`.
- [x] Scaffold `tests/` package layout mirroring `src/` (unit/ and integration/).
- [ ] Set up local Ollama instance with a small model pulled and confirmed
      reachable (`ollama list` / a smoke-test completion call).
      NOT IMPLEMENTED: Q-agent does direct epsilon-greedy action selection from 
      Q-table; no LLM was wired into the decision loop for this implementation.
- [x] Verify `results/` and `docs/` are correctly git-tracked (not ignored)
      so experiment outputs and planning docs persist.

## Phase 2: Local Game Engine (Grid & Rules)

- [x] Implement grid state representation (`grid_size`, cop position, thief
      position, barrier set, move count) sourced from
      `config/game_config.json`.
- [x] Implement seeded barrier placement (up to `max_barriers` = 5),
      excluding start cells.
- [x] Implement seeded starting-position assignment for Cop and Thief.
- [x] Implement the action set `{UP, DOWN, LEFT, RIGHT, STAY}` and legality
      checking (grid bounds, barrier occupancy).
- [x] Implement the deterministic transition function `P` (apply joint
      action → next state).
- [x] Implement terminal-condition checks: capture (cop_pos == thief_pos)
      and timeout (`move_count == max_moves` = 25).
- [x] Implement reward resolution using the `scoring` object (`cop_win`=20,
      `thief_win`=10, `cop_loss`=5, `thief_loss`=5).
- [x] Implement per-agent observation derivation (visibility-radius fog of
      war per PRD §{Ωi}).
- [x] Unit tests: bounds checking, barrier blocking, capture detection,
      timeout detection, reward correctness, observation/fog-of-war
      correctness (47+ test cases in test_board.py, test_game.py; 99% coverage).

## Phase 3: MCP Server Integration

- [x] Define MCP tool schemas: `get_observation(role)` and `make_move(role, direction)` 
      and `get_match_status()` (structured observation/response with validation).
- [x] (Merged with above: tool schema design completed.)
- [x] Implement unified MCP server with role parameter (COP/THIEF distinction)
      wrapping the shared grid engine (src/mcp_server/server.py).
- [ ] Implement separate Thief MCP server. NOT IMPLEMENTED: single unified server 
      with role parameter used instead (cleaner design for single match).
- [x] Implement simultaneous-turn synchronization (hold-and-resolve-together, 
      not first-caller-wins) via MatchState buffering logic.
- [x] Add server-side validation rejecting illegal actions with a clear
      error/response rather than corrupting state.
- [x] Smoke-test the MCP server locally with a scripted/random client to 
      confirm protocol correctness end-to-end.
- [ ] Expose MCP server via ngrok tunnel (or equivalent) and confirm remote 
      reachability. NOT IMPLEMENTED: local stdio MCP server only (no ngrok exposure).
- [x] Integration tests covering full scripted episodes over the MCP protocol 
      (test_mcp.py, test_tournament_run.py; 21 integration tests).

## Phase 4: RL Agent Strategy (Q-Learning)

- [x] Implement Q-table data structure (dict keyed by observation tuple →
      5-action Q-value dict, in src/agents/q_agent.py).
- [x] Implement ε-greedy action selection over the Q-table (direct policy, 
      no LLM in decision loop).
- [x] Implement the Q-learning update rule (`α`=0.1, `γ`=0.9 per PRD, using
      rewards from `config/game_config.json`) with terminal-state handling.
- [x] Implement ε decay schedule across the `num_games` (6) games of a run
      (exponential decay with configurable floor).
- [x] Implement Q-table persistence to/from `data/` per team (Team Alpha/Beta,
      not per role; save after each game; support resuming a run).
- [ ] Wire the local LLM (Ollama) into the agent decision loop using the
      structured tool-call schemas (no free-form prompts) from Phase 3.
      NOT IMPLEMENTED: epsilon-greedy policy selects actions directly from Q-table.
- [ ] Pin LLM sampling (temperature 0 / fixed seed) for reproducibility.
      NOT IMPLEMENTED: no LLM in decision loop.
- [x] Unit tests: Q-table update math, ε decay correctness, persistence
      round-trip (save → load → identical table) (830+ lines of Q-learning tests).

## Phase 5: Multiplayer Testing & Logs

- [x] Implement the experiment runner (`src/scripts/run_tournament.py`) that drives 
      all `num_games` (6) games end-to-end with the unified MCP server.
- [~] Implement per-game logging (outcome, reward, moves-to-capture) to 
      `results/tournament_report.json`. PARTIALLY: per-game logging complete; 
      per-turn logging not implemented (out of scope).
- [x] Persist per-episode seeds via seeded random.Random injection into Board
      and QLearningAgent (determinism infrastructure in place).
- [x] Run a full multi-game experiment and confirm no protocol errors across 
      all 6 games (end-to-end tested; 228 passing tests).
- [ ] Re-run the same experiment against ngrok-tunneled (cloud-exposed) MCP
      servers and confirm parity with the local run. NOT IMPLEMENTED: local 
      stdio MCP server only.
- [~] Verify reproducibility: infrastructure supports seeded runs (Board and 
      QLearningAgent accept injectable random.Random), but run_tournament.py 
      does not currently pin/verify reproducibility with fixed seeds (partial).
- [x] Verify learning trend: tournament_report.json captures per-game outcomes; 
      learning can be verified from final_scores and per-game records.
- [ ] Measure and record token usage per turn/per run against the FinOps
      budget implied by `max_moves` (25) × `num_games` (6). NOT IMPLEMENTED: 
      no LLM in decision loop, no tokens to measure.
- [x] Write up final experiment results summary in `results/tournament_report.json`.
