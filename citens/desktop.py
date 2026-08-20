"""Desktop entry point for the single-exe build (PyInstaller).

Doubles as the ``citens-desktop`` console script in normal installs. The
exe is meant to feel like a desktop app:

    double-click CiteLens.exe
      -> the local web console starts and the browser opens
      -> first run (no API key yet): the console AUTO-OPENS the settings
         page — fill in your LLM provider there; no terminal prompts

Portability rule: EVERYTHING lives next to the exe — .env, .cache, papers/,
runs/, data/. Copy the folder to another machine and it just works; delete
it and nothing is left behind. The console window IS the server: closing it
stops the app.

No ``citens`` import may happen at module top: settings load ``.env`` from
the working directory at import time, and the working directory is only
switched to the exe's folder inside :func:`main`.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _app_dir() -> Path:
    """The portable app folder: the exe's directory (cwd in dev mode)."""
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def _free_port(preferred: int = 8000) -> int:
    """First bindable port from `preferred` upward (8000-8009)."""
    for port in range(preferred, preferred + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def _import_selfcheck() -> str:
    """Verify heavy optional toolchains actually load inside the exe."""
    import importlib

    problems = []
    for mod in ("citens.api.app", "markitdown", "citens.grounding.fulltext"):
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 - report, don't crash
            problems.append(f"{mod}: {e}")
    return "\n".join(problems) or "all imports OK"


def main() -> None:
    # portable mode BEFORE any citens import (settings read .env from cwd).
    # CITELENS_WORKDIR in .env redirects the data directory (runs/, papers/,
    # .cache/, lit pools) — "one copy of the exe, data where I choose it".
    app_dir = _app_dir()
    os.chdir(app_dir)
    from citens.api.envstore import read_env_value

    workdir = read_env_value(app_dir / ".env", "CITELENS_WORKDIR")
    if workdir:
        wd = Path(workdir)
        if not wd.is_absolute():
            wd = app_dir / wd
        wd.mkdir(parents=True, exist_ok=True)
        os.chdir(wd)
        # the config moves WITH the data: everything except the pointer line
        # migrates into the workdir's .env on first use — settings UI, pydantic
        # Settings, and this bootstrap all read/write cwd/.env afterwards
        pointer = app_dir / ".env"
        target = wd / ".env"
        if pointer.is_file() and not target.is_file():
            keep = [
                line for line in pointer.read_text(encoding="utf-8").splitlines()
                if not line.strip().startswith("CITELENS_WORKDIR")
            ]
            target.write_text("\n".join(keep) + "\n", encoding="utf-8")

    if "--import-check" in sys.argv:  # exe smoke test / support tooling
        print(_import_selfcheck())
        return

    import uvicorn

    from citens.api.app import app

    first_run = not read_env_value(Path.cwd() / ".env", "LLM_API_KEY")
    port = _free_port()
    url = f"http://127.0.0.1:{port}" + ("/?setup=1" if first_run else "")
    print()
    print(f"  CiteLens 控制台运行中 / console running: {url}")
    print(f"  工作目录 / workdir: {Path.cwd()}")
    if first_run:
        print("  首次运行：浏览器将打开设置页，填写模型服务商与 API Key 即可")
    print("  保持本窗口开启（关闭窗口 = 退出软件）· Ctrl+C 退出")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\n  bye")


if __name__ == "__main__":
    main()
