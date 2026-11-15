"""Loaded by pytest before every test file."""

# onnxruntime must be imported before pandas/mlflow on Windows (otherwise its DLL crashes),
# so we import it here, first. On Linux (CI, Docker) the order does not matter.
import onnxruntime  # noqa: F401  # isort: skip

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    """A fresh empty folder per test, deleted afterwards.

    Replaces pytest's built-in tmp_path, which also creates a "current" symlink: on some Windows
    machines a symlink attempt waits ~30 s before failing, which made every test 30 s slower.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        yield Path(folder)
