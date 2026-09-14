"""The replay viewer with its Traits tab: serves viz/ and plays a trained Study 2 village with the traits set there.

    uv run python scripts/viewer_server.py              # then open http://127.0.0.1:8000
    uv run python scripts/viewer_server.py --port 8001

The Traits tab needs the trained Study 2 grid in ../hamlet-artefacts/runs/v2/grid and the world it was trained in,
v2-lite (commit 659bc80), which HEAD's economy cannot express: the server re-runs itself on that commit's code,
extracted once from git into ../hamlet-artefacts/code/v2lite_659bc80.

GET  /api/meta?eval_seed=10000   trait caps and neutral values, each named population as drawn for that day set,
                                 and the trained A-S1-N9 runs
POST /api/run                    {"seed", "eval_seed", "name", "traits": [one per agent: appetite, metabolism,
                                 chronotype, laziness, aptitude [farm, office], learning_rate]}
                                 -> {"id": the new replay, "summary": days 1-5, neutral against these traits}

One run at a time, evaluation only, bound to localhost. Episodes go to ../hamlet-artefacts/runs/v2/traits/, replays
to viz/replays/; the grid is only read. Aptitudes are used as set (not normalised), unlike a sampled population.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
ART = REPO.parent / "hamlet-artefacts"
V2LITE = ART / "code" / "v2lite_659bc80"
G = ART / "runs" / "v2" / "grid"
OUT = ART / "runs" / "v2" / "traits"
VIZ = REPO / "viz"

if os.environ.get("V2LITE") != str(V2LITE):
    if not (V2LITE / "hamlet").exists():
        V2LITE.mkdir(parents=True, exist_ok=True)
        tree = subprocess.run(["git", "-C", str(REPO), "archive", "659bc80"], check=True, stdout=subprocess.PIPE).stdout
        subprocess.run(["tar", "-x", "-C", str(V2LITE)], input=tree, check=True)
    env = {**os.environ, "V2LITE": str(V2LITE), "PYTHONPATH": os.pathsep.join([str(V2LITE), str(V2LITE / "scripts")])}
    os.chdir(tempfile.gettempdir())       # so selection.json's relative paths cannot resolve outside the grid
    # -P keeps this file's directory, HEAD's scripts/, off the front of sys.path
    os.execve(sys.executable, [sys.executable, "-P", str(Path(__file__).resolve()), *sys.argv[1:]], env)

import numpy as np                                                                 # noqa: E402
import pandas as pd                                                                # noqa: E402

import hamlet.evaluate as E                                                        # noqa: E402
import hamlet.traits as T                                                          # noqa: E402
from hamlet.config import CANTEEN, FARM, HOME, MARKET, OFFICE, SOCIAL, TRANSIT, ZONE_NAMES  # noqa: E402
from hamlet.core import HamletCore                                                 # noqa: E402
from evaluate_grid import run_config, selected_checkpoint                          # noqa: E402

for module in (E, sys.modules["evaluate_grid"]):
    assert Path(module.__file__).is_relative_to(V2LITE), f"HEAD code loaded instead of v2-lite: {module.__file__}"

SHOWN = ["mean_D", "mean_E", "mean_F", "mean_C", "frac_HOME", "frac_SOCIAL", "frac_TRANSIT", "frac_JOBS",
         "bar_visits_per_day", "copresence_at_bar"]
LOCK = threading.Lock()
NEUTRAL: dict[tuple[int, int], dict] = {}


def dashboard(df: pd.DataFrame, first: int, last: int) -> dict[str, float]:
    """Zone shares of agent-ticks, state means, bar visits a day and company per bar tick over days first..last."""
    d = df[(df["day"] >= first) & (df["day"] <= last)]
    share = d["zone"].value_counts(normalize=True)
    row = {f"frac_{ZONE_NAMES[z]}": float(share.get(z, 0.0)) for z in (HOME, CANTEEN, SOCIAL, MARKET, TRANSIT)}
    row["frac_JOBS"] = float(share.get(FARM, 0.0) + share.get(OFFICE, 0.0))
    row.update({f"mean_{c}": float(d[c].mean()) for c in ("D", "E", "F", "C", "W")})
    visits, company, n_days = [], [], d["day"].nunique()
    for _, g in d.sort_values("t").groupby("agent"):
        at = g["zone"].to_numpy() == SOCIAL
        visits.append(int(np.sum(at & ~np.r_[False, at[:-1]])) / n_days)
        if at.any():
            company.append(float(g["company_others"].to_numpy()[at].mean()))
    row["bar_visits_per_day"] = float(np.mean(visits))
    row["copresence_at_bar"] = float(np.mean(company)) if company else 0.0
    return row


def as_dict(t: T.Traits) -> dict:
    return {"appetite": t.appetite, "metabolism": t.metabolism, "chronotype": t.chronotype, "laziness": t.laziness,
            "aptitude": list(t.aptitude), "learning_rate": t.learning_rate}


def look(e: float, f: float, c: float) -> str:
    """One word for how a village is doing, the same rule as the viewer's: its lowest need if below 0.2, else thriving."""
    word, low = min((("exhausted", e), ("hungry", f), ("lonely", c)), key=lambda x: x[1])
    return word if low < 0.2 else "thriving"


def meta(eval_seed: int) -> dict:
    cfg = run_config(G / "A-S1-N9" / "seed1")
    pc = pd.read_csv(G / "reports" / "v2_grid_per_condition.csv")
    runs = [{"seed": int(r.seed), "checkpoint": r.checkpoint, "state": "escaped" if r.mean_D < 0.2 else "stuck",
             "look": look(r.mean_E, r.mean_F, r.mean_C)}
            for r in pc[pc.condition == "A-S1-N9"].sort_values("seed").itertuples()]
    return {"caps": T.CAPS, "neutral": as_dict(T.Traits.neutral()), "n_agents": cfg.N, "runs": runs,
            "eval_seeds": [E.EVAL_SEED_BASE + k for k in range(E.N_EVAL_SEEDS)],
            "populations": {p: [as_dict(t) for t in E.run_traits(dataclasses.replace(cfg, population=p), eval_seed)]
                            for p in T.POPULATIONS}}


def episode(cfg, ckpt, eval_seed: int, condition: str, traits=None):
    """One evaluation episode through hamlet.evaluate.rollout; `traits` (TraitArrays) replaces the population draw."""
    class Core(HamletCore):
        def __init__(self, c, seed=None, traits_seed=None, **_):
            super().__init__(c, seed, traits_seed=traits_seed, traits=traits)

    policy, label = E.make_policy(fallback=str(ckpt))
    saved, E.HamletCore = E.HamletCore, Core           # under LOCK: rollout builds its core from this name
    try:
        return E.rollout(cfg, policy, eval_seed, condition=condition, checkpoint=label), label
    finally:
        E.HamletCore = saved


def run(body: dict) -> dict:
    seed, eval_seed = int(body["seed"]), int(body["eval_seed"])
    agents = [T.Traits(float(a["appetite"]), float(a["metabolism"]), float(a["chronotype"]), float(a["laziness"]),
                       tuple(float(x) for x in a["aptitude"]), float(a["learning_rate"])) for a in body["traits"]]
    arrays = T.TraitArrays.from_traits(agents).validate()           # ValueError outside the caps
    name = re.sub(r"[^a-z0-9_-]+", "-", str(body.get("name") or "").lower()).strip("-")[:40]
    name = name or "custom-" + hashlib.sha1(json.dumps(body["traits"], sort_keys=True).encode()).hexdigest()[:6]
    run_dir = G / "A-S1-N9" / f"seed{seed}"
    cfg, ckpt = run_config(run_dir), selected_checkpoint(run_dir)
    assert ckpt.resolve().is_relative_to(G), ckpt
    if len(agents) != cfg.N:
        raise ValueError(f"traits for {len(agents)} agents; the village has {cfg.N}")
    if (seed, eval_seed) not in NEUTRAL:
        df0, _ = episode(cfg, ckpt, eval_seed, cfg.condition_name)
        NEUTRAL[(seed, eval_seed)] = dashboard(df0, cfg.burn_in_days, cfg.n_days - 1)
    condition = f"A-S1-N9-traits-{name}"
    df, label = episode(cfg, ckpt, eval_seed, condition, arrays)
    home = OUT / "A-S1-N9" / f"seed{seed}"
    home.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "metadata.json", home / "metadata.json")   # the replay export reads the world from it
    parquet = E.write_log(df, home / "eval" / "ui" / condition / label / f"seed{eval_seed}_ep0.parquet")
    (parquet.parent / f"traits_seed{eval_seed}.json").write_text(json.dumps(body["traits"], indent=2))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "V2LITE")}
    out = subprocess.run(["uv", "run", "python", "scripts/export_replay.py", str(parquet)], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True).stdout
    changed = dashboard(df, cfg.burn_in_days, cfg.n_days - 1)
    return {"id": re.search(r"wrote viz/replays/(\S+)\.json", out).group(1),
            "summary": {"neutral": {k: NEUTRAL[(seed, eval_seed)][k] for k in SHOWN}, "changed": {k: changed[k] for k in SHOWN}}}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIZ), **kwargs)

    def end_headers(self):
        if self.path.startswith("/replays/"):
            self.send_header("Cache-Control", "no-store")         # a re-run under the same name replaces the file
        super().end_headers()

    def _json(self, code: int, obj) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path != "/api/meta":
            return super().do_GET()
        try:
            self._json(200, meta(int(parse_qs(url.query).get("eval_seed", [E.EVAL_SEED_BASE])[0])))
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self.send_error(404)
        if not LOCK.acquire(blocking=False):
            return self._json(409, {"error": "a run is already going; wait for it to finish"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._json(200, run(body))
        except (ValueError, KeyError, TypeError) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            LOCK.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    port = ap.parse_args().port
    print(f"viewer on http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
