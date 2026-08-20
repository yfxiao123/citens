"""CiteLens — a literature-review agent that writes critical, citation-grounded surveys."""

from __future__ import annotations

try:  # installed package metadata tracks pyproject (the release version)
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("citens")
except PackageNotFoundError:  # frozen/source-tree fallback
    __version__ = "1.2.3"

__all__ = ["__version__"]
