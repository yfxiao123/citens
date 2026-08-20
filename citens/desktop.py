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
import time
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
    # IMMEDIATELY: the frozen app spends 30-180s importing (self-extraction +
    # AV scan + heavy first import) — a blank window looks like a hang
    print("CiteLens 正在加载 / loading…", flush=True)
    print("首次运行约 1-3 分钟（自解压 + 杀毒扫描），请勿关闭窗口", flush=True)
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
    base = f"http://127.0.0.1:{port}"
    url = base + ("/?setup=1" if first_run else "")
    print()
    print(f"  CiteLens 控制台 / console: {url}")
    print(f"  工作目录 / workdir: {Path.cwd()}")
    if first_run:
        print("  首次运行：就绪后自动打开设置页，填写模型服务商与 API Key 即可")
    print("  保持本窗口开启（关闭窗口 = 退出软件）· Ctrl+C 退出")

    def _open_when_ready(timeout_s: int = 300) -> None:
        # a frozen exe's first launch spends 30-90s in self-extraction +
        # antivirus scanning BEFORE the server binds — opening the browser
        # early showed ERR_CONNECTION_REFUSED. Gate on /health instead.
        import urllib.request

        t0 = time.monotonic()
        print("  正在启动 / starting", end="", flush=True)
        while time.monotonic() - t0 < timeout_s:
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        print(
                            f"\n  就绪 / ready ({time.monotonic() - t0:.0f}s)"
                            f" — 打开 / opening {url}",
                            flush=True,
                        )
                        webbrowser.open(url)
                        return
            except Exception:  # noqa: BLE001 - not ready yet
                pass
            print(".", end="", flush=True)
            time.sleep(1.0)
        print(f"\n  ⚠ {timeout_s}s 内未就绪；服务仍在启动，请稍后手动访问 {url}")

    threading.Thread(target=_open_when_ready, daemon=True).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\n  bye")


if __name__ == "__main__":
    main()
