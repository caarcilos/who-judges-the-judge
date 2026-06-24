from __future__ import annotations

import os
from pathlib import Path

from .base import RunnerError


def load_env_file(path: str | Path) -> None:
    """Load simple KEY=VALUE entries without overriding existing environment values."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise RunnerError(
                f"{env_path}:{line_number}: expected an environment assignment"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise RunnerError(f"{env_path}:{line_number}: environment key is empty")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RunnerError(
            f"{name} is not set. Copy .env.example to .env and add your value."
        )
    return value

