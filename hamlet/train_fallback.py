"""Single-file PPO over a batch of simulation cores: the project's trainer.

Clipped PPO with the settings of :class:`hamlet.train_common.PPOHyperparameters`,
written directly in torch.

Batching
--------
``BatchedCore`` holds ``B`` independent :class:`hamlet.core.HamletCore`
instances seeded ``env_seed * 1000 + b`` (world streams) and all drawing
their traits from ``env_seed`` (``traits_seed``, recorded once in
``traits.csv``), and steps them in a Python loop of length ``B``; each core
is itself vectorised over its ``N`` agents. The
per-tick cost of the core is flat in ``N`` and about 0.2 ms, so at ``B=16``
the loop is not the bottleneck and a true ``(B, N)`` vectorised core was not
built. Observations are ``(B, N, obs_dim)`` and flattened to ``(B*N, obs_dim)``
for the network; the one-hot agent identity under symmetry S1 is already
part of the observation written by the core, so the network needs no extra
input.

Truncation
----------
Episodes only ever truncate at ``episode_ticks``. When a core reaches that
tick its final observation is evaluated by the critic and used as the
bootstrap value for that transition, the core is reset, and the advantage
recursion is cut there; rollout boundaries that are not episode boundaries
bootstrap from the value of the next observation as usual.

Objective
---------
Clipped surrogate with ``clip_param`` on decision ticks only, value loss
``0.5 * max((V - R)^2, (V_clipped - R)^2)`` with ``V_clipped = V_old +/-
vf_clip_param`` (PPO2 value clipping) on every tick, entropy
bonus ``entropy_coeff``, Adam with ``adam_eps``, gradient-norm clip
``grad_clip``; advantages are
standardised over the whole batch before the epochs. No KL term.

Outputs, in ``<out>/<condition>/seed<s>/``: ``metadata.json``,
``progress.csv`` (one row per update), ``ckpt_NNNN.pt`` every
``checkpoint_every_updates`` updates plus the final one, and
``selection.json``. A checkpoint is loaded for evaluation by
:class:`hamlet.policies.FallbackPolicy`.

``python -m hamlet.train_fallback --smoke`` runs one small update end to end.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from hamlet.config import N_ACTIONS, HamletConfig
from hamlet.core import HamletCore
from hamlet.evaluate import add_config_arguments, config_from_args
from hamlet.metrics.stats import iqm
from hamlet.train_common import (
    CheckpointRecord,
    PPOHyperparameters,
    ProgressLog,
    finish_metadata,
    policy_seed,
    run_directory,
    write_metadata,
    write_selection,
    write_traits,
)

TRAINER_NAME = "fallback"
# Spacing of per-core environment seeds: core b of env seed s uses s * stride + b.
ENV_SEED_STRIDE = 1000
# Orthogonal-initialisation gains: hidden layers, policy head, value head.
HIDDEN_GAIN, POLICY_GAIN, VALUE_GAIN = float(np.sqrt(2.0)), 0.01, 1.0
# Default number of cores per batch and torch threads.
DEFAULT_N_ENVS = 16
DEFAULT_THREADS = 2
# Energy log: one row per run, appended to <repo root>/emissions.csv by codecarbon (optional dependency).
EMISSIONS_FILE = "emissions.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]


def start_emissions_tracker(project_name: str):
    """Start a codecarbon tracker for one run, or return ``None`` when it must not or cannot run.

    Process-level tracking (``tracking_mode="process"``) attributes the CPU
    term to this process alone, so concurrent runs do not each report the
    whole machine; the GPU is excluded (``gpu_ids=[]``: an idle GPU would be
    charged in full to every run); the RAM term is codecarbon's device
    constant and is recorded but not the per-run figure. ``allow_multiple_runs``
    lets several trainers track at once. The tracker is not started under
    pytest or when ``HAMLET_NO_CODECARBON`` is set, so tests and smoke runs
    never write to the shared ``REPO_ROOT / EMISSIONS_FILE``. Options the
    installed version does not know are dropped rather than passed.
    """
    import os
    import sys

    if os.environ.get("HAMLET_NO_CODECARBON") or "pytest" in sys.modules:
        return None
    try:
        import inspect

        from codecarbon import OfflineEmissionsTracker
        from codecarbon.emissions_tracker import BaseEmissionsTracker
    except ImportError:
        return None
    wanted = {"project_name": project_name, "output_dir": str(REPO_ROOT), "output_file": EMISSIONS_FILE,
              "country_iso_code": "GBR", "tracking_mode": "process", "allow_multiple_runs": True,
              "gpu_ids": [], "log_level": "error", "save_to_file": True}
    known = set(inspect.signature(BaseEmissionsTracker.__init__).parameters) | {"country_iso_code"}
    tracker = OfflineEmissionsTracker(**{k: v for k, v in wanted.items() if k in known})
    tracker.start()
    return tracker


def stop_emissions_tracker(tracker, run_dir: Optional[Path] = None) -> dict[str, Any]:
    """Stop the tracker; return the per-run energy record and write it to ``run_dir/emissions.json``.

    The record (``run_id``, ``duration_s``, ``cpu_energy_kwh``, ``ram_energy_kwh``,
    ``gpu_energy_kwh``, ``energy_kwh``, ``emissions_kg``, ``tracking_mode``) is
    taken from the tracker's final data, so it is complete and per run whatever
    the shared CSV's row alignment is. Empty without a tracker; energy logging
    never fails a run.
    """
    if tracker is None:
        return {}
    try:
        emissions = float(tracker.stop() or 0.0)
        data = getattr(tracker, "final_emissions_data", None)
        values = dict(getattr(data, "values", {}) or {}) if data is not None else {}
        record = {
            "run_id": str(getattr(tracker, "run_id", "")),
            "duration_s": values.get("duration"), "cpu_energy_kwh": values.get("cpu_energy"),
            "ram_energy_kwh": values.get("ram_energy"), "gpu_energy_kwh": values.get("gpu_energy"),
            "energy_kwh": values.get("energy_consumed"), "emissions_kg": emissions,
            "tracking_mode": values.get("tracking_mode"), "cpu_model": values.get("cpu_model"),
            "emissions_file": str(REPO_ROOT / EMISSIONS_FILE),
        }
        if run_dir is not None:
            import json

            (Path(run_dir) / "emissions.json").write_text(json.dumps(record, indent=2, default=str))
        return record
    except Exception as exc:  # noqa: BLE001 - energy logging never fails a run
        return {"error": repr(exc)}
# Smoke settings: cores, updates, epochs.
SMOKE_N_ENVS, SMOKE_UPDATES, SMOKE_EPOCHS = 2, 1, 1
SMOKE_SUBDIR = "smoke-fallback"


def _layer(in_dim: int, out_dim: int, gain: float) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, activation: str, out_gain: float) -> nn.Sequential:
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [_layer(last, h, HIDDEN_GAIN), act()]
        last = h
    layers.append(_layer(last, out_dim, out_gain))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Policy network: observation to categorical logits over the ``N_ACTIONS`` actions."""

    def __init__(self, obs_dim: int, hidden: Sequence[int] = (128, 128), activation: str = "tanh") -> None:
        super().__init__()
        self.net = _mlp(obs_dim, hidden, N_ACTIONS, activation, POLICY_GAIN)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Value network: observation to a scalar state value."""

    def __init__(self, obs_dim: int, hidden: Sequence[int] = (128, 128), activation: str = "tanh") -> None:
        super().__init__()
        self.net = _mlp(obs_dim, hidden, 1, activation, VALUE_GAIN)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class BatchedCore:
    """``B`` independent cores stepped together; resets each core at truncation."""

    def __init__(self, cfg: HamletConfig, seeds: Sequence[int], traits_seed: Optional[int] = None) -> None:
        self.cfg = cfg
        self.cores = [HamletCore(cfg, int(s), traits_seed=traits_seed) for s in seeds]
        self.B = len(self.cores)
        self.N = cfg.N
        self.episode_returns: list[float] = []     # mean over agents, one per finished episode
        self._running = np.zeros((self.B, self.N), dtype=np.float64)

    def reset(self) -> np.ndarray:
        """Reset every core (continuing each core's stream) and return ``(B, N, obs_dim)``."""
        self._running[:] = 0.0
        return np.stack([core.reset() for core in self.cores])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step every core with ``actions (B, N)``.

        Returns ``(next_obs (B, N, obs_dim), rewards (B, N), truncated (B,),
        final_obs (B, N, obs_dim))``. ``next_obs`` is the observation the
        policy acts on next (the post-reset observation for a truncated
        core); ``final_obs`` is the last observation of the episode where
        ``truncated`` is set and equals ``next_obs`` elsewhere.
        """
        next_obs = np.empty((self.B, self.N, self.cfg.obs_dim), dtype=np.float32)
        final_obs = np.empty_like(next_obs)
        rewards = np.empty((self.B, self.N), dtype=np.float32)
        truncated = np.zeros(self.B, dtype=bool)
        for b, core in enumerate(self.cores):
            obs, rew, info = core.step(actions[b])
            rewards[b] = rew
            final_obs[b] = obs
            self._running[b] += rew
            if info["truncated"]:
                truncated[b] = True
                self.episode_returns.append(float(self._running[b].mean()))
                self._running[b] = 0.0
                obs = core.reset()
            next_obs[b] = obs
        return next_obs, rewards, truncated, final_obs

    def decision_mask(self) -> np.ndarray:
        """``bool (B, N)``: whether the next action given to :meth:`step` will be used by each core."""
        return np.stack([core.can_decide for core in self.cores])

    def mean_drive(self) -> float:
        """Mean drive over every core and agent, as a snapshot of the current state."""
        return float(np.mean([core.D.mean() for core in self.cores]))

    def pop_episode_returns(self) -> list[float]:
        out, self.episode_returns = self.episode_returns, []
        return out


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    done: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation over ``(T, M)`` arrays.

    ``next_values[t]`` is the bootstrap value of the state after transition
    ``t`` (the truncated episode's final observation when ``done[t]``);
    ``done`` cuts the recursion. Returns ``(advantages, returns)``.
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    last = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + values


def save_checkpoint(
    path: Path,
    actor: Actor,
    critic: Critic,
    cfg: HamletConfig,
    hyper: PPOHyperparameters,
    update: int,
    agent_steps: int,
) -> Path:
    """Write the state dicts plus enough metadata to rebuild the networks."""
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "obs_dim": cfg.obs_dim,
            "hidden": list(hyper.hidden),
            "activation": hyper.activation,
            "hamlet_config": asdict(cfg),
            "hyperparameters": asdict(hyper),
            "update": update,
            "agent_steps": agent_steps,
        },
        path,
    )
    return path


def load_actor(path: Path) -> tuple[Actor, dict[str, Any]]:
    """Rebuild the actor from a checkpoint; returns ``(actor in eval mode, checkpoint dict)``."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    actor = Actor(int(ckpt["obs_dim"]), tuple(ckpt["hidden"]), ckpt.get("activation", "tanh"))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, ckpt


def ppo_update(
    actor: Actor,
    critic: Critic,
    optimiser: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    hyper: PPOHyperparameters,
    rng: torch.Generator,
) -> dict[str, float]:
    """Run ``num_epochs`` passes of minibatch gradient steps; returns mean loss terms."""
    n = batch["obs"].shape[0]
    adv = batch["adv"]
    adv = (adv - adv.mean()) / torch.clamp(adv.std(), min=1e-4)
    stats = {"policy_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0, "approx_kl": 0.0}
    steps = 0
    for _ in range(hyper.num_epochs):
        order = torch.randperm(n, generator=rng)
        for start in range(0, n, hyper.minibatch_size):
            idx = order[start : start + hyper.minibatch_size]
            # The policy is only responsible for the ticks whose action it actually chose; on the
            # ticks of a running journey the action was held, so the ratio there carries no signal.
            # The critic predicts every tick, so the value loss keeps the whole minibatch.
            mask = batch["decision"][idx]
            weight = mask.sum().clamp(min=1.0)
            dist = Categorical(logits=actor(batch["obs"][idx]))
            logp = dist.log_prob(batch["actions"][idx])
            ratio = torch.exp(logp - batch["logp"][idx])
            a = adv[idx]
            surrogate = torch.min(ratio * a, torch.clamp(ratio, 1 - hyper.clip_param, 1 + hyper.clip_param) * a)
            policy_loss = -(surrogate * mask).sum() / weight
            entropy = (dist.entropy() * mask).sum() / weight
            v = critic(batch["obs"][idx])
            returns = batch["returns"][idx]
            # PPO2-style value clipping: the clip bounds how far the prediction may move from the
            # one made at rollout time, so it never zeroes the gradient at this return scale
            # (a clamp on the squared error did, on 76% of samples).
            v_old = batch["vf_old"][idx]
            v_clipped = v_old + torch.clamp(v - v_old, -hyper.vf_clip_param, hyper.vf_clip_param)
            vf_loss = torch.max((v - returns) ** 2, (v_clipped - returns) ** 2).mean()
            loss = policy_loss + hyper.vf_loss_coeff * vf_loss - hyper.entropy_coeff * entropy
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), hyper.grad_clip)
            optimiser.step()
            with torch.no_grad():
                stats["policy_loss"] += float(policy_loss)
                stats["vf_loss"] += float(vf_loss)
                stats["entropy"] += float(entropy)
                stats["clip_fraction"] += float((((ratio - 1).abs() > hyper.clip_param).float() * mask).sum() / weight)
                stats["approx_kl"] += float(((batch["logp"][idx] - logp) * mask).sum() / weight)
            steps += 1
    return {k: v / max(steps, 1) for k, v in stats.items()}


def train(
    cfg: HamletConfig,
    hyper: PPOHyperparameters,
    env_seed: int,
    out_dir: Path,
    n_envs: int = DEFAULT_N_ENVS,
    max_updates: Optional[int] = None,
    threads: int = DEFAULT_THREADS,
    verbose: bool = True,
    codecarbon: bool = True,
) -> Path:
    """Train one run and return its directory.

    ``hyper.train_batch_size`` must equal ``rollout_ticks * n_envs * N``; the
    rollout length in ticks is derived from it. Stops after
    ``hyper.total_agent_steps`` agent-steps or ``max_updates`` updates. With
    ``codecarbon`` (and the package installed) one energy row per run is
    appended to ``<repo root>/emissions.csv``, tagged with the run directory,
    and the per-run record is written to ``<run_dir>/emissions.json``.
    ``metadata.json`` is completed at the end with ``finished_utc``,
    ``wall_clock_s``, ``agent_steps``, ``updates`` and the energy record.
    """
    cfg.validate()
    torch.set_num_threads(threads)
    n_agents = cfg.N
    if hyper.train_batch_size % (n_envs * n_agents) != 0:
        raise ValueError("train_batch_size must be a multiple of n_envs * n_agents")
    T = hyper.train_batch_size // (n_envs * n_agents)
    M = n_envs * n_agents

    run_dir = run_directory(out_dir, cfg, env_seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(run_dir, cfg, hyper, TRAINER_NAME, env_seed,
                   extra={"n_envs": n_envs, "rollout_ticks": T, "threads": threads, "traits_seed": int(env_seed)})
    write_traits(run_dir, cfg, env_seed)
    log = ProgressLog(run_dir / "progress.csv")

    torch.manual_seed(policy_seed(env_seed))
    shuffle_rng = torch.Generator().manual_seed(policy_seed(env_seed))
    actor = Actor(cfg.obs_dim, hyper.hidden, hyper.activation)
    critic = Critic(cfg.obs_dim, hyper.hidden, hyper.activation)
    optimiser = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=hyper.lr, eps=hyper.adam_eps)

    envs = BatchedCore(cfg, [env_seed * ENV_SEED_STRIDE + b for b in range(n_envs)], traits_seed=env_seed)
    obs = envs.reset().reshape(M, cfg.obs_dim)
    tracker = start_emissions_tracker(str(run_dir)) if codecarbon else None   # tagged by run directory

    buf_obs = np.zeros((T, M, cfg.obs_dim), dtype=np.float32)
    buf_act = np.zeros((T, M), dtype=np.int64)
    buf_logp = np.zeros((T, M), dtype=np.float32)
    buf_val = np.zeros((T, M), dtype=np.float32)
    buf_next_val = np.zeros((T, M), dtype=np.float32)
    buf_rew = np.zeros((T, M), dtype=np.float32)
    buf_done = np.zeros((T, M), dtype=np.float32)
    buf_dec = np.zeros((T, M), dtype=np.float32)      # 1 where the policy's action is actually used

    records: list[CheckpointRecord] = []
    returns_since_ckpt: list[float] = []
    agent_steps = 0
    update = 0
    start = time.perf_counter()
    while agent_steps < hyper.total_agent_steps and (max_updates is None or update < max_updates):
        t_sample = time.perf_counter()
        drive_sum, drive_n = 0.0, 0     # reset each update: mean_D is this batch's mean
        with torch.no_grad():
            for t in range(T):
                obs_t = torch.from_numpy(obs)
                dist = Categorical(logits=actor(obs_t))
                actions = dist.sample()
                buf_dec[t] = envs.decision_mask().reshape(M)     # read before stepping
                drive_sum += envs.mean_drive(); drive_n += 1
                buf_obs[t] = obs
                buf_act[t] = actions.numpy()
                buf_logp[t] = dist.log_prob(actions).numpy()
                buf_val[t] = critic(obs_t).numpy()
                next_obs, rew, trunc, final_obs = envs.step(buf_act[t].reshape(n_envs, n_agents))
                buf_rew[t] = rew.reshape(M)
                done = np.repeat(trunc, n_agents).astype(np.float32)
                buf_done[t] = done
                obs = next_obs.reshape(M, cfg.obs_dim)
                # Bootstrap from the final observation of a truncated episode, else from the next state.
                boot = np.where(done[:, None] > 0, final_obs.reshape(M, cfg.obs_dim), obs)
                buf_next_val[t] = critic(torch.from_numpy(np.ascontiguousarray(boot))).numpy()
        adv, ret = compute_gae(buf_rew, buf_val, buf_next_val, buf_done, hyper.gamma, hyper.gae_lambda)
        sample_time = time.perf_counter() - t_sample

        t_learn = time.perf_counter()
        batch = {
            "obs": torch.from_numpy(buf_obs.reshape(T * M, cfg.obs_dim)),
            "actions": torch.from_numpy(buf_act.reshape(T * M)),
            "logp": torch.from_numpy(buf_logp.reshape(T * M)),
            "adv": torch.from_numpy(adv.reshape(T * M).astype(np.float32)),
            "returns": torch.from_numpy(ret.reshape(T * M).astype(np.float32)),
            "decision": torch.from_numpy(buf_dec.reshape(T * M)),
            "vf_old": torch.from_numpy(buf_val.reshape(T * M).astype(np.float32)),
        }
        stats = ppo_update(actor, critic, optimiser, batch, hyper, shuffle_rng)
        learn_time = time.perf_counter() - t_learn

        update += 1
        agent_steps += T * M
        finished = envs.pop_episode_returns()
        returns_since_ckpt.extend(finished)
        row = {
            "update": update,
            "agent_steps": agent_steps,
            "episodes": len(finished),
            "episode_return_iqm": iqm(np.asarray(finished)) if finished else float("nan"),
            "episode_return_mean": float(np.mean(finished)) if finished else float("nan"),
            "mean_reward": float(buf_rew.mean()),
            # A mean over every tick of the batch. It used to be a snapshot of the cores taken
            # after the last tick, which on the one update in six that ended on an episode
            # boundary reported the drive of freshly reset agents (~0.25) instead of the batch's
            # own (~2.0), and so read as a periodic dip that never happened.
            "mean_D": drive_sum / max(drive_n, 1),
            **stats,
            "sample_s": sample_time,
            "learn_s": learn_time,
            "elapsed_s": time.perf_counter() - start,
        }
        log.write(row)
        if verbose:
            print(f"update {update:4d}  steps {agent_steps:>9d}  return {row['episode_return_mean']:9.2f}  "
                  f"entropy {stats['entropy']:.3f}  {sample_time:.1f}s+{learn_time:.1f}s")

        if update % hyper.checkpoint_every_updates == 0:
            path = save_checkpoint(run_dir / f"ckpt_{update:04d}.pt", actor, critic, cfg, hyper, update, agent_steps)
            records.append(CheckpointRecord(str(path), update, agent_steps, returns_since_ckpt))
            returns_since_ckpt = []

    if not records or records[-1].update != update:
        path = save_checkpoint(run_dir / f"ckpt_{update:04d}.pt", actor, critic, cfg, hyper, update, agent_steps)
        records.append(CheckpointRecord(str(path), update, agent_steps, returns_since_ckpt))
    energy = stop_emissions_tracker(tracker, run_dir)
    wall = time.perf_counter() - start
    finish_metadata(run_dir, {"wall_clock_s": wall, "agent_steps": agent_steps, "updates": update,
                              "agent_steps_per_s": agent_steps / max(wall, 1e-9),
                              "emissions_kg": energy.get("emissions_kg"), "energy": energy or None})
    write_selection(run_dir, records)
    return run_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--seed", type=int, default=0, help="environment seed s; policy seed is 100 + s")
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS, help="cores stepped per update")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="torch CPU threads")
    parser.add_argument("--updates", type=int, default=None, help="stop after this many updates")
    parser.add_argument("--agent-steps", type=int, default=None, help="total agent-steps budget")
    parser.add_argument("--entropy-coeff", type=float, default=None)
    parser.add_argument("--reward-form", default=None, choices=["level", "difference"])
    parser.add_argument("--out", default="runs")
    parser.add_argument("--smoke", action="store_true", help=f"two cores, one small update into <out>/{SMOKE_SUBDIR}")
    parser.add_argument("--no-codecarbon", action="store_true", help="do not log energy to emissions.csv")
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    if args.reward_form is not None:
        cfg.reward_form = args.reward_form
    hyper = PPOHyperparameters()
    if args.agent_steps is not None:
        hyper = replace(hyper, total_agent_steps=args.agent_steps)
    if args.entropy_coeff is not None:
        hyper = replace(hyper, entropy_coeff=args.entropy_coeff)
    n_envs, max_updates, out = args.n_envs, args.updates, Path(args.out)
    if args.smoke:
        n_envs, max_updates, out = SMOKE_N_ENVS, SMOKE_UPDATES, Path(args.out) / SMOKE_SUBDIR
        batch = cfg.ticks_per_day * n_envs * cfg.N
        hyper = replace(hyper, train_batch_size=batch, minibatch_size=batch // 2,
                        num_epochs=SMOKE_EPOCHS, checkpoint_every_updates=1)

    start = time.perf_counter()
    run_dir = train(cfg, hyper, args.seed, out, n_envs=n_envs, max_updates=max_updates, threads=args.threads,
                    codecarbon=not args.no_codecarbon)
    print(f"finished {run_dir} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
