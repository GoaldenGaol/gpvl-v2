"""Tests for package and data path resolution."""

from pathlib import Path

from volition.paths import data_file, package_root, project_root


def test_package_root_exists():
    root = package_root()
    assert (root / "__init__.py").exists()


def test_project_root_has_pyproject():
    root = project_root()
    assert (root / "pyproject.toml").exists()


def test_bundled_data_file():
    path = data_file("dim4_frozen_2023.csv")
    assert path.exists()
    assert path.suffix == ".csv"


def test_all_datasets_resolvable():
    for name in (
        "dim4_frozen_2023.csv",
        "dim4_frozen_2023_full.csv",
        "dim4_firms_frozen_2023.csv",
    ):
        assert data_file(name).exists()