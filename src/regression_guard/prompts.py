"""Loading and resolving versioned prompts from /prompts."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import PromptConfig


def load_prompt(path: str | Path) -> PromptConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"prompt file {path} must contain a YAML mapping")
    return PromptConfig.model_validate(data)


def list_prompts(prompt_dir: str | Path) -> list[Path]:
    return sorted(Path(prompt_dir).glob("*.yaml"))


def resolve_prompt(prompt_dir: str | Path, ref: str | None) -> PromptConfig:
    """Resolve a prompt by file path, bare version id, or 'latest'."""
    candidates = list_prompts(prompt_dir)
    if not candidates:
        raise FileNotFoundError(f"no prompt files in {prompt_dir}")

    if ref in (None, "", "latest"):
        loaded = [load_prompt(p) for p in candidates]
        return max(loaded, key=lambda c: (c.created_at, c.version))

    as_path = Path(ref)
    if as_path.exists():
        return load_prompt(as_path)

    for path in candidates:
        config = load_prompt(path)
        if config.version == ref or path.stem == ref:
            return config

    known = ", ".join(p.stem for p in candidates)
    raise FileNotFoundError(f"no prompt matching {ref!r}. Available: {known}")
