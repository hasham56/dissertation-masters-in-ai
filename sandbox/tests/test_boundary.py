"""Import direction between the sandbox and the study, and the shape of the sandbox modules.

The sandbox imports ``hamlet`` as a read-only library; nothing under
``hamlet/``, ``tests/`` or ``scripts/`` imports ``sandbox``. Run with
``uv run pytest sandbox/tests``.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
IMPORT = re.compile(r"^\s*(from|import)\s+([\w.]+)", re.M)


def _imports(folder: Path, prefix: str) -> list[str]:
    hits = []
    for path in folder.rglob("*.py"):
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        for match in IMPORT.finditer(path.read_text(errors="ignore")):
            module = match.group(2)
            if module == prefix or module.startswith(prefix + "."):
                hits.append(f"{path.relative_to(ROOT)}: {match.group(0).strip()}")
    return hits


def test_study_scripts_and_tests_never_import_sandbox():
    for folder in ("hamlet", "tests", "scripts"):
        assert _imports(ROOT / folder, "sandbox") == []


def test_sandbox_outputs_are_gitignored():
    assert "sandbox/runs/" in (ROOT / ".gitignore").read_text().splitlines()


def test_sandbox_env_subclasses_the_core_and_overrides_only_the_reward_side():
    from hamlet.core import HamletCore

    core = importlib.import_module("sandbox.core")
    assert issubclass(core.SandboxEnv, HamletCore)
    own = {name for name, value in vars(core.SandboxEnv).items() if callable(value)}
    assert own == {"__init__", "_drive", "step", "lever_summary"}
    for name in ("sandbox.train", "sandbox.fingerprint", "sandbox.readout"):
        importlib.import_module(name)


def test_dry_run_validates_every_config_without_training(capsys):
    train = importlib.import_module("sandbox.train")
    for path in sorted((ROOT / "sandbox" / "configs").glob("*.json")):
        train.main([str(path), "--dry-run"])
        out = capsys.readouterr().out
        assert '"lever_summary"' in out and '"exploratory"' not in out.split('"config"')[0]
    with pytest.raises(SystemExit):
        train.main([])
