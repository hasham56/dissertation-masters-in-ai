"""E1, exploratory: does a learned agent use SOCIAL when someone is reliably there?

    uv run python scripts/exploratory_e1.py --root runs --out runs/exploratory/E1

The confirmatory grid found social exactly zero in every A-S1 cell: the agents never visit SOCIAL
at all. Social credit needs another agent active in the same zone at the same tick, so a solo trip
pays nothing and the gradient points away from the only action that could ever pay. E1 asks whether
that is the whole story by removing the coordination problem rather than the need: the last two agents
are replaced by GREEDY-CLOCK, which does visit SOCIAL, and the remaining learned agents keep their
own policy and see SOCIAL occupancy through the observation exactly as before.

Exploratory throughout. Nothing here enters a confirmatory family, no threshold is applied, and
only dashboard quantities are computed: time fractions, need levels and occupancy counts.

Writes ``<out>/e1.csv``, ``<out>/e1.md`` and one Parquet per (cell, evaluation seed) under
``<out>/<condition>/seed<s>/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamlet.config import SOCIAL, HamletConfig  # noqa: E402
from hamlet.evaluate import episode_rng, evaluation_seeds, frame_from_columns  # noqa: E402
from hamlet.config import LOG_COLUMNS  # noqa: E402
from hamlet.core import HamletCore  # noqa: E402
from hamlet.policies import FallbackPolicy, GreedyClockPolicy  # noqa: E402

# The last two agent indices are replaced by GREEDY-CLOCK; the rest keep the learned policy. At
# N = 8 this was the literal (6, 7); deriving it from N keeps the arm's definition, two scripted
# neighbours, when the population moves.
SCRIPTED_AGENTS = tuple(range(HamletConfig().n_agents - 2, HamletConfig().n_agents))


class MixedPolicy:
    """Learned policy for most agents, GREEDY-CLOCK for ``scripted``.

    Both policies see the whole observation and both are asked for every agent; only their own
    rows are kept. That keeps each policy's internal behaviour exactly as it is when it runs alone,
    and it keeps the world stream identical to an unmixed rollout.
    """

    name = "MIXED"

    def __init__(self, learned, scripted_ids: Sequence[int]) -> None:
        self.learned = learned
        self.scripted = GreedyClockPolicy()
        self.ids = np.asarray(sorted(scripted_ids), dtype=np.int64)

    def act(self, obs: np.ndarray, core: HamletCore):
        a_learned, logps = self.learned.act(obs, core)
        a_scripted, _ = self.scripted.act(obs, core)
        actions = np.asarray(a_learned, dtype=np.int64).copy()
        actions[self.ids] = np.asarray(a_scripted, dtype=np.int64)[self.ids]
        return actions, logps


def rollout_mixed(cfg: HamletConfig, policy, seed: int, condition: str, checkpoint: str) -> pd.DataFrame:
    """One episode with a mixed population; mirrors hamlet.evaluate.rollout tick for tick."""
    core = HamletCore(cfg, seed, traits_seed=seed)
    core.rng = episode_rng(seed, 0)
    obs = core.reset(None)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in LOG_COLUMNS}
    held = None
    for _ in range(cfg.episode_ticks):
        decision = core.can_decide
        if decision.any():
            actions, logps = policy.act(obs, core)
            if logps is not None:
                lp = np.asarray(logps, dtype=np.float32)
                held = lp if held is None else np.where(decision[:, None], lp, held)
        else:
            actions = np.zeros(cfg.N, dtype=np.int64)
        obs, rewards, _ = core.step(actions)
        cols = core.log_columns(seed, condition, checkpoint, 0, actions, rewards, held)
        for name in LOG_COLUMNS:
            columns[name].append(cols[name])
    return frame_from_columns(columns)


def summarise(df: pd.DataFrame, cfg: HamletConfig, learned_ids: np.ndarray) -> dict[str, float]:
    """Dashboard quantities for the learned agents only, plus the occupancy they could see."""
    d = df[df["day"] >= cfg.burn_in_days]
    mine = d[d["agent"].isin(learned_ids)]
    at_social = mine["zone"] == SOCIAL
    # ticks on which somebody (anybody) was active at SOCIAL, from the whole population
    busy = d[(d["zone"] == SOCIAL) & d["active"]].groupby(["seed", "t"]).size()
    busy_keys = set(map(tuple, busy.index.to_frame().to_numpy())) if len(busy) else set()
    keys = list(map(tuple, mine[["seed", "t"]].to_numpy()))
    with_company = np.fromiter((k in busy_keys for k in keys), dtype=bool, count=len(keys))
    return {
        "learned_social_frac": float(at_social.mean()),
        "learned_social_active_ticks": int((at_social & mine["active"]).sum()),
        "learned_mean_C": float(mine["C"].mean()),
        "learned_mean_D": float(mine["D"].mean()),
        "frac_ticks_at_social_while_occupied": float((at_social.to_numpy() & with_company).mean()),
        "any_social_occupied_frac": float(with_company.mean()),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="runs")
    parser.add_argument("--out", default="runs/exploratory/E1")
    parser.add_argument("--condition", default=HamletConfig().condition_name)
    parser.add_argument("--n-eval-seeds", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="training seeds (default: all found)")
    parser.add_argument("--keep-parquet", action="store_true", help="write every episode log")
    args = parser.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_seeds = evaluation_seeds(args.n_eval_seeds)
    cells = sorted((root / args.condition).glob("seed*/selection.json"))
    if args.seeds is not None:
        cells = [c for c in cells if int(c.parent.name.removeprefix("seed")) in set(args.seeds)]
    if not cells:
        print(f"no finished cells under {root / args.condition}")
        return

    rows = []
    for sel_path in cells:
        run = sel_path.parent
        seed = int(run.name.removeprefix("seed"))
        meta = json.loads((run / "metadata.json").read_text())
        cfg = HamletConfig(**meta["hamlet_config"])
        ckpt = run / Path(json.loads(sel_path.read_text())["selected"]).name
        learned_ids = np.array([i for i in range(cfg.N) if i not in SCRIPTED_AGENTS], dtype=np.int64)
        print(f"{args.condition} seed {seed}: {ckpt.name}, agents {list(SCRIPTED_AGENTS)} scripted")

        for arm, make in (("mixed", lambda: MixedPolicy(FallbackPolicy(ckpt), SCRIPTED_AGENTS)),
                          ("unmixed", lambda: FallbackPolicy(ckpt))):
            frames = []
            for es in eval_seeds:
                df = rollout_mixed(cfg, make(), es, cfg.condition_name, ckpt.stem)
                frames.append(df)
                if args.keep_parquet:
                    p = out / args.condition / f"seed{seed}" / arm / f"seed{es}_ep0.parquet"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(p, engine="pyarrow", index=False)
            all_df = pd.concat(frames, ignore_index=True)
            rows.append({"condition": args.condition, "seed": seed, "arm": arm,
                         "n_eval_seeds": len(eval_seeds), **summarise(all_df, cfg, learned_ids)})
            print(f"    {arm:8s} social {rows[-1]['learned_social_frac']:.5f}  "
                  f"active {rows[-1]['learned_social_active_ticks']:6d}  "
                  f"C {rows[-1]['learned_mean_C']:.4f}  D {rows[-1]['learned_mean_D']:.4f}")

    t = pd.DataFrame(rows)
    t.to_csv(out / "e1.csv", index=False)
    piv = t.pivot_table(index="seed", columns="arm",
                        values=["learned_social_frac", "learned_social_active_ticks",
                                "learned_mean_C", "frac_ticks_at_social_while_occupied"])
    md = [
        "# E1: a learned population with two scripted neighbours (exploratory)",
        "",
        "**Exploratory.** Outside every confirmatory family, no threshold applied, dashboard "
        "quantities only.",
        "",
        f"Agents {list(SCRIPTED_AGENTS)} of each A-S1 cell are replaced by GREEDY-CLOCK, which does "
        "visit SOCIAL; the six learned agents keep their own policy and observe SOCIAL occupancy as "
        "usual. Every number below counts only the learned agents, over days 1-5 of "
        f"{len(eval_seeds)} evaluation episodes. `unmixed` is the same checkpoint with no "
        "substitution, as the comparison.",
        "",
        t.to_markdown(index=False),
        "",
        "## Per seed, mixed against unmixed",
        "",
        piv.round(5).to_markdown(),
        "",
    ]
    (out / "e1.md").write_text("\n".join(md) + "\n")
    print(f"\n{len(t)} row(s) -> {out / 'e1.csv'} and {out / 'e1.md'}")


if __name__ == "__main__":
    main()
