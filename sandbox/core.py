"""``SandboxEnv``: ``hamlet.core.HamletCore`` with the sandbox reward-design levers.

What is overridden, and nothing else
------------------------------------
``HamletCore`` is used unchanged for the world: movement, contention, opening
hours, state deltas, traits, gossip, shocks, observations and logging. The
subclass overrides exactly two methods:

``_drive``
    The study computes ``D = w_E (1 - E)^2 + w_F (1 - F)^2 + w_C (1 - C)^2``
    with ``w`` the ``(3,)`` config weights (``core.py`` 472-475) and the set
    point ``1.0`` as a literal. ``SandboxEnv._drive`` computes
    ``D = W[:, 0] dev_E + W[:, 1] dev_F + W[:, 2] dev_C`` with ``S`` and ``W``
    ``(N, 3)`` arrays (population-wide values broadcast, per-agent overrides
    written into rows) and ``dev_i = (s_i - x_i)^2`` (``set_point_form=
    "symmetric"``) or ``max(0, s_i - x_i)^2`` (``"deficit"``). The expression
    keeps the study's operation order term by term, so with every ``s_i = 1``
    and ``W`` equal to the config weights the result is bit-identical to
    ``HamletCore._drive`` (``sandbox/tests/test_parity.py``). Because
    ``HamletCore`` calls ``_drive`` at reset (250) and in step 4 (431), the
    override changes the reward, the mood trace and the logged ``D`` column
    together. ``self.weights`` (the config ``(3,)`` vector the scripted
    schedulers read) is left as ``HamletCore`` sets it.

``step``
    Calls ``HamletCore.step`` and, when ``collective_lambda > 0``, returns
    ``(1 - lambda) r + lambda * mean(r)`` in place of the reward vector; the
    raw vector is kept in ``last_raw_reward`` and the returned one in
    ``last_shaped_reward`` so a rollout can log the raw reward and write the
    shaped one to a sidecar. States, ``D`` and mood are untouched. The mixing
    is sum-preserving: ``sum((1 - l) r + l mean(r)) = (1 - l) sum(r) + l N mean(r) = sum(r)``.

Levers honoured (config keys validated by ``load_config`` in ``sandbox/train.py``): ``set_points``,
``set_points_per_agent``, ``set_point_form`` (the set-point lever); population-wide
``drive_weights`` through ``HamletConfig.w_*`` and ``drive_weights_per_agent``
here (the weight lever); ``population`` / ``traits_seed`` / ``traits_override`` through
``HamletCore``'s own arguments (the trait lever); ``collective_lambda`` (the collective-reward lever). Not honoured
by design: every world lever (there is no argument for them),
the arm and the symmetry (they come from ``base``).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from hamlet.config import HamletConfig
from hamlet.core import HamletCore
from hamlet.traits import TraitArrays

NEEDS = ("energy", "satiety", "social")          # order of the E, F, C columns and of the weight vector
SET_POINT_FORMS = ("symmetric", "deficit")


def need_array(
    population_wide: Optional[Mapping[str, float]],
    per_agent: Optional[Mapping[Any, Mapping[str, float]]],
    n_agents: int,
    default: float,
) -> np.ndarray:
    """Build the ``(N, 3)`` array of a per-need lever from population-wide values and per-agent overrides.

    ``population_wide`` maps need names to values (missing needs take
    ``default``); ``per_agent`` maps agent indices (ints or their decimal
    strings) to partial mappings that overwrite single entries of that
    agent's row. Unknown need names and agent indices outside
    ``range(n_agents)`` raise ``ValueError``.
    """
    base = np.full(3, float(default), dtype=np.float64)
    for name, value in (population_wide or {}).items():
        if name not in NEEDS:
            raise ValueError(f"unknown need {name!r}; choose from {NEEDS}")
        base[NEEDS.index(name)] = float(value)
    out = np.tile(base, (int(n_agents), 1))
    for key, partial in (per_agent or {}).items():
        try:
            agent = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"agent index {key!r} is not an integer") from exc
        if not 0 <= agent < n_agents:
            raise ValueError(f"agent index {agent} outside range({n_agents})")
        for name, value in partial.items():
            if name not in NEEDS:
                raise ValueError(f"unknown need {name!r} for agent {agent}; choose from {NEEDS}")
            out[agent, NEEDS.index(name)] = float(value)
    return out


class SandboxEnv(HamletCore):
    """``HamletCore`` with set points, per-agent drive weights and the collective reward term."""

    def __init__(
        self,
        cfg: HamletConfig,
        seed: Optional[int] = None,
        traits_seed: Optional[int] = None,
        traits: Optional[TraitArrays] = None,
        set_points: Optional[Mapping[str, float]] = None,
        set_points_per_agent: Optional[Mapping[Any, Mapping[str, float]]] = None,
        set_point_form: str = "symmetric",
        drive_weights_per_agent: Optional[Mapping[Any, Mapping[str, float]]] = None,
        collective_lambda: float = 0.0,
    ) -> None:
        """Construct the world exactly as ``HamletCore`` does, then install the levers.

        ``cfg`` carries the population-wide drive weights (``w_energy``,
        ``w_satiety``, ``w_social``) and the trait population; ``traits`` (an
        explicit ``TraitArrays``) is the ``traits_override`` path. The lever
        arrays are built before ``HamletCore.__init__`` runs, because its
        ``reset`` calls ``_drive``. Validation: set points in ``(0, 1]``,
        weights ``>= 0``, ``set_point_form`` in ``SET_POINT_FORMS``,
        ``collective_lambda`` in ``[0, 1]``; anything else raises
        ``ValueError`` and never clamps.
        """
        cfg.validate()
        n = cfg.N
        if set_point_form not in SET_POINT_FORMS:
            raise ValueError(f"unknown set_point_form {set_point_form!r}; choose from {SET_POINT_FORMS}")
        lam = float(collective_lambda)
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"collective_lambda must lie in [0, 1]; got {lam}")
        points = need_array(set_points, set_points_per_agent, n, 1.0)
        if np.any(points <= 0.0) or np.any(points > 1.0):
            raise ValueError(f"set points must lie in (0, 1]; got {points.tolist()}")
        config_weights = {"energy": cfg.w_energy, "satiety": cfg.w_satiety,
                          "social": cfg.w_social if cfg.social_on else 0.0}
        weights = need_array(config_weights, drive_weights_per_agent, n, 1.0)
        if np.any(weights < 0.0):
            raise ValueError(f"drive weights must be non-negative; got {weights.tolist()}")
        self._set_points = points
        self._lever_weights = weights
        self._set_point_form = set_point_form
        self._lambda = lam
        self._per_agent_set_points = dict(set_points_per_agent or {})
        self._per_agent_weights = dict(drive_weights_per_agent or {})
        self._explicit_traits = traits is not None
        self.last_raw_reward: np.ndarray = np.zeros(n, dtype=np.float32)
        self.last_shaped_reward: np.ndarray = np.zeros(n, dtype=np.float32)
        super().__init__(cfg, seed, traits_seed, traits)

    # ------------------------------------------------------------------ levers
    def _drive(self) -> np.ndarray:
        """``D = W_E dev_E + W_F dev_F + W_C dev_C``, ``(N,)``; see the module docstring."""
        s = self._set_points
        w = self._lever_weights
        d_e = s[:, 0] - self.E
        d_f = s[:, 1] - self.F
        d_c = s[:, 2] - self.C
        if self._set_point_form == "deficit":
            d_e = np.maximum(d_e, 0.0)
            d_f = np.maximum(d_f, 0.0)
            d_c = np.maximum(d_c, 0.0)
        return w[:, 0] * d_e ** 2 + w[:, 1] * d_f ** 2 + w[:, 2] * d_c ** 2

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """``HamletCore.step`` followed by the collective mixing of the reward vector."""
        obs, rew, info = super().step(actions)
        self.last_raw_reward = rew
        if self._lambda > 0.0:
            rew = ((1.0 - self._lambda) * rew + self._lambda * rew.mean()).astype(np.float32)
        self.last_shaped_reward = rew
        return obs, rew, info

    def lever_summary(self) -> dict[str, Any]:
        """The resolved lever state (set points, weights, form, lambda, trait source) for ``metadata.json``."""
        cfg = self.cfg
        return {
            "set_points": dict(zip(NEEDS, self._set_points[0].tolist())) if self._per_agent_set_points == {} else None,
            "set_points_matrix": self._set_points.tolist(),
            "set_points_per_agent": {str(k): dict(v) for k, v in self._per_agent_set_points.items()},
            "set_point_form": self._set_point_form,
            "drive_weights": {"energy": cfg.w_energy, "satiety": cfg.w_satiety,
                              "social": cfg.w_social if cfg.social_on else 0.0},
            "drive_weights_matrix": self._lever_weights.tolist(),
            "drive_weights_per_agent": {str(k): dict(v) for k, v in self._per_agent_weights.items()},
            "collective_lambda": self._lambda,
            "population": cfg.population,
            "traits_seed": int(self.traits_seed),
            "traits_source": "explicit" if self._explicit_traits else f"population {cfg.population!r}",
            "neutral": bool(
                np.all(self._set_points == 1.0)
                and np.all(self._lever_weights == np.array([cfg.w_energy, cfg.w_satiety,
                                                            cfg.w_social if cfg.social_on else 0.0]))
                and self._lambda == 0.0
            ),
        }
