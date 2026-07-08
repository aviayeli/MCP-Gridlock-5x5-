# PLAN: MCP-Gridlock-5x5 Implementation

This document lays out the pragmatic build plan for the system defined in
`docs/PRD.md`: two MCP servers (Cop, Thief) fronting a shared 5x5 grid game,
each driven by a locally-hosted small LLM whose actions are informed by a
persisted Q-table.

## 1. Technical Approach

### 1.1 Component layout

```
src/
  game/        # authoritative grid engine: state, transitions, legality
  mcp_servers/ # Cop server + Thief server (tool definitions, validation)
  agents/      # local-LLM-driven policy wrapper around Ollama + Q-table
  rl/          # Q-table representation, update rule, persistence
  cli/         # experiment runner: drives num_games episodes end-to-end
```

The grid engine (`game/`) is the single source of truth for legality and
transitions; both MCP servers import it rather than re-implementing rules,
so Cop and Thief always agree on what a legal move is.

### 1.2 Q-table representation & persistence

- **Key**: each agent's Q-table is keyed by its own **observation**, not the
  raw global state — consistent with the Dec-POMDP design in the PRD (fog of
  war beyond the visibility radius). Concretely, a key is a tuple:
  `(self_pos, opponent_pos_or_None, barrier_layout_id, move_bucket)`.
  `opponent_pos` is `None` when the opponent is outside the visibility
  radius, which collapses many true states into one Q-table entry and keeps
  the table small.
- **Value**: a length-5 array of Q-values, one per action in
  `{UP, DOWN, LEFT, RIGHT, STAY}`.
- **Representation**: a flat `dict[key, np.ndarray(5,)]` (or an equivalent
  plain-Python dict of lists if avoiding a numpy dependency) — the state
  space is small enough (bounded by 25 positions × 25 opponent-or-none ×
  barrier layouts × move buckets) that a dense table or nested arrays are
  unnecessary; a dict keeps sparse/unvisited states implicitly at a default
  value.
- **Update rule**: standard tabular Q-learning,
  `Q[s,a] += α (r + γ · max_a' Q[s',a'] − Q[s,a])`, with `γ = 0.9` (per PRD)
  and a fixed learning rate `α` (e.g. 0.1) and an ε-greedy exploration
  schedule that decays across the 6 games in a run so later games lean more
  on learned behavior.
- **Persistence**: each agent's Q-table is serialized to
  `results/<agent>_qtable.json` (or `.pkl`) after every game, keyed by a
  JSON-safe string encoding of the tuple key. Persisting after every game
  (not just at the end of the run) means a crashed run can resume from the
  last completed game rather than losing all learning.

### 1.3 MCP tool schema design

Two tools per server, minimal and structured (per the FinOps requirement in
the PRD — no free-form fields):

- `get_state()` → returns the calling agent's observation object:
  `{self_pos: [x,y], opponent_pos: [x,y] | null, barriers: [[x,y], ...],
  move_count: int, moves_remaining: int}`.
- `submit_action(action: "UP"|"DOWN"|"LEFT"|"RIGHT"|"STAY")` → server
  validates legality (bounds + barrier check via the shared `game/` engine),
  applies the transition once **both** agents have submitted for the current
  turn, and returns `{accepted: bool, new_state: <observation>, done: bool,
  reward: int | null}`.

Turn synchronization (server-side): the shared game state advances only once
per completed turn-pair; a server holds a submitted action until its
counterpart server (or a shared state store) confirms the other agent has
also submitted, then both transitions resolve together per `P`. This keeps
turns simultaneous rather than giving whichever agent calls first an
information/tempo advantage.

### 1.4 Barrier placement

- `max_barriers` (5) barriers are placed once per episode (not per turn), by
  a deterministic seeded procedure: given an episode seed, sample `k ≤ 5`
  non-adjacent-to-start cells (excluding the Cop's and Thief's starting
  cells) uniformly at random from the 5x5 grid using a seeded RNG.
- The same episode seed is recorded in the episode log so a barrier layout
  is fully reproducible from the seed alone — no need to store the raw
  layout separately, though it is stored anyway for convenience/debugging.

### 1.5 Determinism & reproducibility (for grading)

Three independent sources of randomness must be pinned to get a
byte-identical replay:

1. **Barrier layout seed** (per episode) — as above.
2. **Starting positions seed** (per episode) — Cop/Thief start cells drawn
   the same way.
3. **LLM sampling** — the local model is called with `temperature=0` (or the
   Ollama equivalent, e.g. a fixed seed parameter) so the same observation
   always yields the same action given the same Q-table snapshot. Since the
   grid transition function `P` is deterministic (per PRD), pinning these
   three sources is sufficient for full run reproducibility: same seeds +
   same starting Q-tables ⇒ same sequence of states, observations, actions,
   and rewards.

All three seeds are logged per episode in `results/` alongside the game log,
so any run can be replayed exactly for debugging or grading.

## 2. Milestones

1. **Grid engine** — implement `game/` (state, legal-move checking,
   deterministic transition, terminal-condition checks) with unit tests
   covering bounds, barrier blocking, and capture/timeout resolution.
2. **MCP servers (stubbed policy)** — stand up Cop and Thief MCP servers
   with `get_state`/`submit_action` tools backed by the grid engine, driven
   first by a random or scripted policy (no LLM yet) to validate the
   protocol and turn synchronization.
3. **Local LLM agent wrapper** — wire an Ollama-backed small model into the
   agent loop, using the structured schemas from §1.3, replacing the
   scripted policy.
4. **Q-learning integration** — add the Q-table lookup/update/persistence
   layer (§1.2) so agent decisions are ε-greedy over learned Q-values
   instead of purely LLM free-choice, and rewards from `config/
  game_config.json`'s `scoring` object flow back into updates.
5. **Cloud exposure** — front each MCP server with an ngrok tunnel (or
   equivalent) so the two agents can run as genuinely separate endpoints,
   confirming the architecture in PRD §3 works over real network hops, not
   just localhost.
6. **Full experiment run & logging** — run all `num_games` (6) games
   end-to-end, capturing per-turn logs, per-game outcomes/rewards, and
   final Q-table snapshots into `results/`.

## 3. Key Design Decisions (summary)

| Decision | Choice | Why |
|---|---|---|
| Q-table key | Own observation (not global state) | Matches Dec-POMDP partial observability; keeps table small |
| Turn order | Simultaneous, server-synchronized | Avoids first-mover information advantage |
| Barriers | Fixed per episode, seeded | Deterministic + reproducible without storing full layouts redundantly |
| LLM sampling | Temperature 0 / fixed seed | Required for reproducible grading |
| Tool schemas | Fixed structured JSON, 2 tools/server | Minimizes tokens per turn (FinOps) |
| Reward source | Directly from `config/game_config.json` scoring | Single source of truth, no duplicated constants |
