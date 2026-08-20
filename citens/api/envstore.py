"""Tiny .env reader/writer for the settings API and the desktop entry.

The desktop app treats ``.env`` next to the exe (or in CITELENS_WORKDIR) as
its config store. This module does line-level KEY=VALUE handling that
preserves comments and unknown keys verbatim — the file stays human-editable
with a text editor, the settings UI just rewrites known lines.
"""

from __future__ import annotations

from pathlib import Path


def env_path() -> Path:
    """The active .env (working directory — set by desktop.py before import)."""
    return Path.cwd() / ".env"


def read_env_value(path: Path, key: str) -> str | None:
    """One key's raw value from an env file (None if absent/empty)."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        if k.strip() == key:
            v = v.strip().strip('"').strip("'")
            return v or None
    return None


def update_env_file(path: Path, updates: dict[str, str]) -> None:
    """Set/update keys in an env file, preserving every other line verbatim.

    Keys whose value is empty/None are removed (unset). Missing file is
    created with a small header.
    """
    lines: list[str] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    done: set[str] = set()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        key = s.partition("=")[0].strip() if "=" in s and not s.startswith("#") else ""
        if key in updates:
            if key in done:
                continue  # drop duplicate definitions, keep the last write
            done.add(key)
            val = updates[key].strip()
            if val:
                out.append(f"{key}={val}")
            # empty value -> key removed entirely
        else:
            out.append(line)
    for key, val in updates.items():
        if key not in done and val.strip():
            out.append(f"{key}={val.strip()}")
    while out and not out[-1].strip():
        out.pop()
    header = "" if out and out[0].startswith("#") else (
        "# managed by the CiteLens settings UI — edit freely, unknown lines are kept\n"
    )
    path.write_text(header + "\n".join(out) + "\n", encoding="utf-8")
