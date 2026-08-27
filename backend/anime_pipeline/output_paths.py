from __future__ import annotations

from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path("./output")

_RUN_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT


def set_run_output_root(output_root: str | Path) -> Path:
    """Set the base directory used for all generated artifacts in this run."""
    global _RUN_OUTPUT_ROOT
    _RUN_OUTPUT_ROOT = Path(output_root).expanduser()
    return _RUN_OUTPUT_ROOT


def get_run_output_root() -> Path:
    """Return the current run-scoped output root."""
    return _RUN_OUTPUT_ROOT


def run_output_path(*parts: str | Path) -> Path:
    """Resolve a path inside the current run-scoped output root."""
    return _RUN_OUTPUT_ROOT.joinpath(*parts)
