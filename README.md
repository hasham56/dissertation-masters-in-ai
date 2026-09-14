# Hamlet

A homeostatic multi-agent village. Nine agents carry three internal states (energy, satiety and a
social need), earn coins at a farm or an office, buy food at a market stall or a restaurant, drink
at a bar and rest at home. One parameter-shared PPO policy drives every agent, each rewarded for
keeping its own drive low. The repository trains that policy, evaluates it on fixed seeds against
four hand-written schedulers, computes routine, specialisation and co-presence metrics with
permutation nulls, and plays any logged episode back on a map.

The simulation core is pure NumPy with a PettingZoo adapter on top, and every
evaluation log is a Parquet file with one row per (tick, agent).

## Setup

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra rl        # simulation, metrics, figures, tests, and torch for training
```

Always sync with `--extra rl`: a bare `uv sync` removes the training extra again.

## Tests

```bash
uv run pytest                    # the study suite, about 40 s
uv run pytest sandbox/tests      # the sandbox suite, about a minute (it trains a short parity run)
```

`tests/test_regression_world.py` pins a hash of every agent's trajectory over two days per
scheduler. A change that is meant to be behaviour-preserving must leave it green; a deliberate world
change regenerates it with `uv run python -m tests.test_regression_world --update`.

## Training

```bash
uv run python -m hamlet.train_fallback --smoke                          # one small update, a few seconds
uv run python -m hamlet.train_fallback --arm A --symmetry S1 --seed 0   # one cell, 30M agent-steps
uv run python scripts/run_grid.py --grid minimum --dry-run              # the 32-cell grid, commands only
uv run python scripts/run_grid.py --conditions A-S1 --seeds 0 1 2 --concurrent 3 --threads 2
```

`run_grid.py` refuses a dirty working tree, a live trainer, a lock, or a finished cell whose settings
hash disagrees with the current code; `--force` overrides all of that and is never used for a study
launch. Cells land under `runs/<condition>/seed<k>/` with their checkpoints, `progress.csv` and a
`metadata.json` that records the exact world they were trained in.

The trainer, `hamlet/train_fallback.py`, is a single-process PPO in torch over a batch of simulation
cores.

## Evaluation, metrics and reports

```bash
uv run python scripts/run_baselines.py --n-seeds 32                     # the four schedulers on the fixed seeds
uv run python scripts/evaluate_grid.py --root runs --passes all --jobs 4 # every pass on every finished cell
uv run python scripts/report_grid.py --root runs --section all          # training table and degeneracy flags
uv run python scripts/compute_metrics.py --root runs --passes all --jobs 4   # metrics with nulls
uv run python scripts/aggregate_manifest.py --root runs                 # runs/manifest.csv with energy totals
uv run python scripts/confirmatory.py --root runs                       # the registered contrasts
uv run python scripts/make_figures.py --root runs                       # figures and results_tables.md
```

`bash scripts/reproduce.sh` runs all of that in order over `runs/` (`ROOT=... JOBS=...` to change
the root and the parallelism). Every step is resumable and skips work whose outputs exist.

A pilot is read with `scripts/report_pilot.py`; the scripted schedulers' competence gate and
calibration tables come from `scripts/report_gate.py` and `scripts/report_clockmi.py`;
`scripts/derive_h1a_threshold.py` derives the H1a threshold by the registered rule. Every number the
analysis relies on is fixed in `scripts/analysis_constants.py`, each with the section of the analysis
plan that registers it.

The pre-registered analysis plan is preserved in this repository's history at tag `analysis-plan-v1` and lives with the project artefacts.

Condition names carry the population size (`A-S1-N9`), and every script that reads a run root
takes the population from that root's own `metadata.json`, never from the current config, so an
archived root trained at another N is read under its own names.

## Running on a cloud VM

`scripts/cloud_bootstrap.sh` prepares a fresh Ubuntu 22.04 machine from a tarball of the tree
(`git archive -o /tmp/hamlet.tgz HEAD`), syncs the environment, runs the suite and measures the
training rate. `scripts/cloud_run.sh` launches a grid under `nohup`, arms `scripts/post_grid.sh`
(evaluation, metrics, reports and the exploratory arms once training ends) and installs
`scripts/cloud_watchdog.sh`, a cron job that deletes the instance after two hours with nothing
running — an in-progress fetch counts as running. `scripts/cloud_fetch.sh` brings a run root home
without overwriting any local cell. Each script's header carries its flags.

## The replay viewer

```bash
uv run python scripts/export_replay.py runs/A-S1-N9/seed0/eval/stochastic/A-S1-N9/ckpt_0864/seed10000_ep0.parquet
uv run python scripts/export_layout.py           # viz/layout.json, from the map's zones layer
cd viz && python3 -m http.server 8000            # then open http://localhost:8000
```

A replay carries the world it was run in — population, zone capacities, printed names, the bar's
floor and slope — read from the run's `metadata.json`; an export with no manifest is refused. The
layout carries positions only, so one map serves every world. The viewer reads Phaser from cdnjs, so
the first load needs a network connection. Each exported Parquet becomes one replay under
`viz/replays/`, and `manifest.json` there lists every replay in the directory.

The tileset images are a licensed asset pack and are not in the repository: place them under
`viz/assets/` (the paths are those the map's tilesets name, `viz/map/hamlet.tmj` lists them) or the
viewer stops and prints the missing path on screen. The agents are drawn from
`viz/assets/characters/character1` to `character9` and the bar's staff from `characters/bartender`,
each folder holding eight 48×48 rotations named by direction (`south.png`, `north-east.png`, …).

`window.arrivalCheck()` in the browser console compares, for every journey in the loaded replay, the
position drawn on the arrival tick with the anchor the agent should stand on. `cd viz && npm install
--no-save puppeteer-core && node verify.mjs http://localhost:8000` runs it over every replay
headlessly and writes screenshots to `viz/shots/`.

## The sandbox

`sandbox/` is an exploratory reward-design workspace. It imports `hamlet` as a read-only library and
subclasses the core to add reward levers; nothing under `hamlet/`, `tests/` or `scripts/` imports
it, and `sandbox/tests/test_boundary.py` enforces that. Its neutral configuration reproduces the
study bit for bit (`sandbox/tests/test_parity.py`). Experiment configs are JSON files under
`sandbox/configs/`, validated by `load_config` in `sandbox/train.py`; `bash sandbox/run_campaign.sh`
runs a set of them. Its outputs go under `sandbox/runs/` and are never confirmatory.

## Layout

```
hamlet/            the simulation, policies, traits, trainers, evaluation and metrics
scripts/           grid launch, evaluation, metrics, reports, figures, cloud, replay export
tests/             the study suite and its pinned trajectory fixture
sandbox/           the reward-design workspace and its suite
viz/               the replay viewer: index.html, the Tiled map, layout.json
```

## Licence

MIT, see `LICENSE`.
