# Hamlet

A homeostatic multi-agent village. Nine agents each carry three needs (energy, satiety and a social
need); they earn coins at a farm or an office, buy food at a market stall or a restaurant, drink at a
bar and rest at home. One parameter-shared PPO policy drives every agent, each rewarded for keeping
its own drive low. This repository trains that policy, evaluates it against four hand-written
schedulers, measures routines, specialisation and co-presence, and plays episodes back on a map.

## Setup

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra rl                # always with --extra rl: a bare `uv sync` removes torch again
uv run pytest                     # the study suite, about 40 s
uv run pytest sandbox/tests       # the sandbox suite, about a minute
```

## Train

```bash
uv run python -m hamlet.train_fallback --smoke                          # one small update, a few seconds
uv run python -m hamlet.train_fallback --arm A --symmetry S1 --seed 0   # one cell, 30M agent-steps
uv run python scripts/run_grid.py --grid minimum --dry-run              # the 32-cell grid, commands only
```

Each run lands in `runs/<condition>/seed<k>/` with its checkpoints and a `metadata.json` recording
the world it was trained in.

## Evaluate and report

```bash
bash scripts/reproduce.sh         # baselines, evaluation, metrics, confirmatory tests and figures over runs/
```

It trains nothing and skips any step whose outputs already exist. Each step is also its own script
in `scripts/`. Every number the analysis relies on is fixed in `scripts/analysis_constants.py`.

The pre-registered analysis plan is preserved in this repository's history at tag `analysis-plan-v1` and lives with the project artefacts.

## Watch the village

```bash
uv run python scripts/viewer_server.py              # then open http://127.0.0.1:8000
uv run python scripts/viewer_server.py --port 8001  # if 8000 is taken
```

- **Replay tab:** pick a replay, press Space to play, click a villager to follow them.
- **Traits tab:** set each villager's appetite, metabolism, bedtime, laziness, job skills and
  learning, then press Run. A trained Study 2 village plays a new episode with those traits, the
  replay opens on the map, and a summary compares days 1 to 5 with the unchanged village.

The Traits tab needs the project artefacts folder next to this repository
(`../hamlet-artefacts/runs/v2/grid`) and commit `659bc80` in the git history, because that village
was trained in the v2-lite world. To watch replays only, `cd viz && python3 -m http.server 8000` is
enough.

To add a replay, export any evaluation log:

```bash
uv run python scripts/export_replay.py runs/A-S1-N9/seed0/eval/stochastic/A-S1-N9/ckpt_0864/seed10000_ep0.parquet
```

The map's tileset images and characters are a licensed asset pack and are not in the repository.
Place them under `viz/assets/`; the viewer names any missing file on screen. Phaser loads from
cdnjs, so the first load needs a network connection.

## Also here

- `sandbox/`: an exploratory reward-design workspace that uses `hamlet` read-only; its results are
  never confirmatory.
- `scripts/cloud_*.sh`: run the grid on a cloud VM; each script's header lists its flags.

## Layout

```
hamlet/     the simulation, policies, traits, trainers, evaluation and metrics
scripts/    grid launch, evaluation, metrics, reports, figures, cloud, replay export, viewer server
tests/      the study suite and its pinned trajectory fixture
sandbox/    the reward-design workspace and its suite
viz/        the viewer: index.html, the Tiled map, layout.json, replays/
```

## Licence

MIT, see `LICENSE`.
