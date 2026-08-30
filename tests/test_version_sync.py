"""The frozen exe reports the __init__ fallback; pip metadata reports
pyproject. The v1.4.0 incident: they drifted (exe said 1.3.3) and the
release had to be retracted. This test fails at gate time on drift."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import citens


def test_fallback_version_matches_package_metadata():
    try:
        meta = version("citens")
    except PackageNotFoundError:  # not installed (bare source tree) — n/a
        return
    assert citens.__version__ == meta, (
        f"__init__ fallback ({citens.__version__}) != pyproject ({meta}) — "
        "bump both together (or the exe will ship reporting the wrong version)"
    )
