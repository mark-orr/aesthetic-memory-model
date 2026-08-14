# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A computational model of aesthetic judgment, built on `pyactup` (an ACT-R-style declarative
memory library). The core idea: a song's memory chunk activation, compared against the
activation of the chunk memory would otherwise predict, is the "kernel" of an aesthetic
signal. The full theoretical framing and every algorithm/experiment spec lives in
**`literature-notes/main-project-idea.txt` — read it fresh at the start of a session**, not
from memory. Mark (the project owner) writes new Algorithm/Design entries there directly,
often as rough, intentionally-vague drafts he wants formalized collaboratively (propose a
concrete interpretation, flag ambiguous parameters, confirm before implementing literally —
don't treat a new entry as a finished spec).

## Commands

```bash
source .venv/bin/activate              # the project venv; not conda, despite .vscode/settings.json's
                                        # default env manager pointing there (conda is broken on this
                                        # machine as of this writing)
pip install -r requirements.txt

python run_simulation.py               # runs run_design1() via __main__

# Execute a notebook non-interactively (regenerates its CSV/figure outputs):
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=aesthetic-memory-model notebooks/<name>.ipynb
```

The registered Jupyter kernel is named `aesthetic-memory-model` (points at `.venv/bin/python`
via a kernelspec installed with `python -m ipykernel install --user --name=aesthetic-memory-model`).
**After editing `src/model.py`, `src/controller.py`, `src/utils.py`, or `run_simulation.py`,
restart any already-running kernel before re-running notebook cells** — Python caches the old
module in memory, so edits won't take effect otherwise (this has bitten this project before).

There is no test suite, linter, or build step configured in this repo.

## Architecture

**`src/model.py`** — `AestheticMemoryModel` wraps a single `pyactup.Memory` and implements a
progression of algorithms, each building on the last:
- **Algorithm A**: `aesthetic_basis(song_id)` — retrieve the chunk memory predicts next
  (`predicted_chunk()`), look up the actual song's own chunk activation
  (`song_activation()`), return `|predicted - actual|`. Chunks: `{song_id, complexity}`.
- **Algorithm A2**: `evaluate_a2(song_id)` — `evaluation = mirrored_sigmoid(y) * (1/e^x)`,
  where `x` is the predicted activation and `y` is the signed diff between a song's own last
  two activation readings. Pure/read-only — leaves chunk-learning to the caller.
- **Algorithm A3/A4**: `evaluate_a3`/`evaluate_a4(song_id)` — `evaluation` = the left-anchored
  inverted parabola (`src/utils.py`) evaluated at `x=aesthetic_basis`, `r`=a running
  time-averaged aesthetic basis (`time_average_mode="cumulative"|"window"`); A4 adds a
  `gamma` scaling factor on `r` (`gamma=1.0` reduces exactly to A3). **Unlike A2, these two
  have a side effect**: they call `learn_evaluation()` internally as a final step. A driver
  loop must not also call it — that double-learns.
- Chunks in A2/A3/A4 are `{song_id, evaluation}`, not `{song_id, complexity}`. Since
  `evaluation` is a continuous float, **every new value creates a new chunk** rather than
  reinforcing an existing one — a song accumulates many chunks over a run. This makes the
  unconstrained "predict next chunk" retrieval scale roughly quadratically with trial count
  (confirmed empirically), unlike Designs 1/2's flat cost — why later designs default to far
  fewer exposures (~2000, or less) than Designs 1/2 (10000).
- `pin_activation(slots, target)` / `adjust_activation(slots, delta)`: force or nudge a
  chunk's activation, both built on `pyactup.Memory.extra_activation` (the only lever pyactup
  exposes for this — there's no direct "set activation" API). They share one callback; a pin
  takes precedence over an adjustment on the same chunk. `pin_activation` requires
  `noise=0` and replicates pyactup's own base-level formula to stay exact across time/
  reinforcement; `adjust_activation` has no such restriction since it's a raw additive offset.
- pyactup has **no way to seed its own internal activation-noise RNG** — only this project's
  own song-sequence sampling is seeded (`seed=` params below). Two separate `noise>0` runs
  with the same `seed` will *not* reproduce each other. This has caused real confusion in
  past comparisons here; use `noise=0` for anything that needs to be strictly reproducible.

**`src/controller.py`** — Algorithm C bandit controllers for Design 6: `SlidingWindowUCB`
(Algorithm C1, non-stationary UCB bandit over the fixed song pool, reward = observed
`evaluation`) and `ChangeAwareUCB` (Algorithm C2, adds privileged access to A4's live `r` —
resets all arms' reward history when `r` shifts more than a relative threshold, to
re-adapt immediately after the inverted-U target moves).

**`src/utils.py`** — `inverted_parabola`/`inverted_parabola_left_anchored` (the latter is the
shape used by Algorithms A3/A4/C; left root fixed at 0, vertex at `r/2`, right root at `r`)
and `compute_convergence_metrics()` (convergence trial + steady-state mean, for Design 6).

**`run_simulation.py`** — the simulation driver: `run_design1()` through `run_design6()`,
each pairing an environment with an algorithm, writing to `results/data/*.csv`:
- Design 1: random ergodic environment (uniform i.i.d. song draws) + Algorithm A.
- Design 2: cyclic von Mises "moving spotlight" environment (`sweep_center`/
  `von_mises_weights`) + Algorithm A.
- Design 3: Design 2's environment + Algorithm A2.
- Design 4/5: Design 1's environment + Algorithm A3/A4 (baseline studies).
- Design 6: Design 1's environment, but song selection comes from a bandit controller
  (`controller="c1"|"c2"`) instead of random draws, + Algorithm A4.

`make_ergodic_environment()` is the shared song-pool generator (fixed complexity per song,
used for reporting even though A2+ don't store complexity in the chunk itself).

**`notebooks/`** — one exploration notebook per Design, plus two preliminary
math-exploration notebooks for the parabola helpers. Each has a parameter cell at the top
(song count, exposures, noise, algorithm-specific knobs) calling the matching `run_designN`,
then plots. This is the primary way Mark interacts with the model day to day, and he often
edits/runs these directly himself outside the conversation — check for uncommitted or
recently-committed notebook changes before assuming you know their current parameter state.

**`docs/`** — static, self-contained HTML/JS/SVG pages published via GitHub Pages (no build
step, no dependencies): `index.html` is an interactive parabola explorer; `design5-demo.html`
/ `design5-demo-2.html` are animated playbacks of pre-computed real simulation runs (not live
re-simulation — pyactup can't run client-side and GitHub Pages has no backend). New demo
pages should follow this same pattern: run the simulation, export a small embedded JSON
frame array, keep prior demo pages intact unless told to replace them, cross-link between
versions in their footers.

**`presentation/`** — generated slide decks (`python-pptx`, since Node/npm and LibreOffice
aren't available on this machine — the usual pptxgenjs skill path doesn't work here). Local
only, gitignored. `literature-notes/presentation.txt` holds the specs for each draft.

**`literature-notes/`** — `main-project-idea.txt` is the living spec (see top of this file);
`*.pdf` files are copyrighted journal articles and are gitignored (this is a public repo).

## Working conventions specific to this project

- **Ask Mark for simulation/model parameter values** (`num_songs`, `num_exposures`, `noise`,
  `gamma`, thresholds, etc.) rather than picking them yourself and justifying from
  convention afterward — his explicit standing preference, established after a rework had
  to be redone over exactly this.
- **Commit locally freely; always ask before `git push`.** Mark sometimes commits directly
  himself (e.g. via VS Code's git UI) — check `git log` for unfamiliar commits rather than
  assuming you have the full picture; it's fine to just ask "expected?"
- pyactup version pinned in `requirements.txt`; when unsure about its exact API behavior,
  introspect the installed source directly (`.venv/lib/python*/site-packages/pyactup.py`)
  rather than trusting docs/memory, which have been stale or paraphrased before.
