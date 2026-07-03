"""Resolve project and package data paths."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path


@lru_cache(maxsize=1)
def package_root() -> Path:
    """Return the installed volition package directory."""
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def project_root() -> Path:
    """
    Return the repository root when running from a source checkout.

    Falls back to the package parent chain for editable installs.
    """
    candidate = package_root().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return package_root().parents[1]


def data_file(filename: str) -> Path:
    """
    Resolve a bundled dim4 CSV by filename.

    Search order:
    1. importlib.resources (installed wheel)
    2. src/volition/data/files/ (editable install)
    3. repo data/ directory (development checkout)
    """
    try:
        traversable = resources.files("volition.data.files").joinpath(filename)
        with resources.as_file(traversable) as bundled:
            if bundled.exists():
                return bundled
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    for base in (
        package_root() / "data" / "files",
        project_root() / "data",
    ):
        path = base / filename
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Dataset '{filename}' not found in package bundle or project data/ directory."
    )


def default_arxiv_dir() -> Path:
    """Default LaTeX output directory relative to project root."""
    return project_root() / "docs" / "arxiv"