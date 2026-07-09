"""The `Policy` interface contract: how a pluggable per-team strategy plugs
into `Tournament`.

Split out from tournament.py purely to respect this package's 150-line-per
-file cap; the policy-callable *contract* is a standalone interface concern
in its own right, separate from both game-driving (tournament.py) and
role-swap scheduling (tournament_schedule.py) — it says nothing about when
or how often a policy is called, only what shape it must have.
"""

from __future__ import annotations

from collections.abc import Callable

from engine.observation import Observation
from engine.player import Action

Policy = Callable[[Observation], Action]
"""Interface contract for a pluggable per-team strategy.

A policy is any callable that takes the single `Observation` its role
currently sees (own position, opponent position or `UNSEEN` if out of the
fog-of-war radius, the static barrier layout, and the current move count)
and returns the `Action` to submit for this turn. Policies are role- and
team-agnostic at the type level — the same callable shape plays Cop in one
game and Thief in the next under the role-swap schedule (see
`tournament_schedule.py`), exactly because `Observation` is already
relative to whichever role holds it. This
is the *only* seam `Tournament` exposes to its caller: a test can inject a
trivial deterministic policy (e.g. always `Action.STAY`, or "step toward
the last-seen opponent"), and a later phase can inject a real LLM- or
Q-table-driven policy, without either side changing anything in this file.
"""
