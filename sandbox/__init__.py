"""Exploratory reward-design workspace. Imports ``hamlet`` as a read-only library.

Nothing in this package is confirmatory. Outputs go under ``sandbox/runs/`` only.
Nothing under hamlet/, tests/ or scripts/ is modified from here; the reward levers are
documented on the config keys that ``load_config`` in ``sandbox/train.py`` validates.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

SANDBOX_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SANDBOX_ROOT.parent
RUNS_DIR = SANDBOX_ROOT / "runs"           # the only permitted output location
EXPLORATORY_TAG = {"exploratory": True}    # written into every metadata file the sandbox produces


def assert_sandbox_output(path: Path) -> Path:
    """Resolve ``path`` and raise ``ValueError`` unless it lies inside ``sandbox/runs/``.

    The parent chain is resolved (the path itself need not exist yet).
    """
    candidate = Path(path)
    resolved = candidate.resolve()
    runs = RUNS_DIR.resolve()
    if resolved != runs and runs not in resolved.parents:
        raise ValueError(f"sandbox outputs must lie inside {RUNS_DIR}; got {candidate}")
    return resolved


def git_describe() -> str:
    """``git describe --always --dirty --tags`` of the repository, or ``"unknown"`` outside a repository."""
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty", "--tags"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
