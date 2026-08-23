"""Desktop entry point for the single-exe build (PyInstaller).

Doubles as the ``citens-desktop`` console script in normal installs. The
exe is meant to feel like a desktop app:

    double-click CiteLens.exe
      -> the local web console starts and the browser opens
      -> first run (no API key yet): the console AUTO-OPENS the settings
         page — fill in your LLM provider there; no terminal prompts
      -> second launch while running: just opens the browser again
         (single instance — no duplicate servers)

The build is WINDOWED (console=False): the onefile bootloader's parent +
child process pair each spawn a terminal window on Windows 11's default
terminal, which users read as "the app opened twice". The web console IS
the UI — there is nothing to read in a terminal. Consequences handled
here: no console streams (NullIO + error.log next to the exe), exit via
the console's ⏻ button (POST /shutdown), and a single-instance probe so
a running app is reused instead of duplicated.

Data location (``_resolve_workdir``): CITELENS_WORKDIR in the exe-folder
.env redirects; otherwise an exe-folder .env means portable-folder mode
(data lives with this copy); otherwise the launch reattaches to the LAST
used workdir (pointer under %LOCALAPPDATA%/CiteLens) so a freshly
downloaded exe never presents an empty workspace; first launch ever uses
the machine home. The header chip in the web console shows the active
workdir.

No ``citens`` import may happen at module top: settings load ``.env`` from
the working directory at import time, and the working directory is only
switched inside :func:`main`.
"""

from __future__ import annotations

import io
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _app_dir() -> Path:
    """The portable app folder: the exe's directory (cwd in dev mode)."""
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd()


class _NullIO(io.TextIOBase):
    """Stand-in for stdout/stderr in windowed builds (no console exists).

    Must be a REAL stream protocol (io.TextIOBase), not a bare write/flush
    stub: uvicorn's ColourizedFormatter calls ``isatty()`` when logging
    boots, and any library may call fileno()/encoding — a minimal stub
    crashed the app on every double-click launch (v1.3.0).
    """

    encoding = "utf-8"
    errors = "replace"

    def write(self, s: str) -> int:  # noqa: ARG002
        return len(s)

    def writable(self) -> bool:
        return True


def _console_safe() -> None:
    """Windowed builds have no console: sys.stdout/stderr may be None and
    any print() would raise. Install a null sink; console builds keep their
    real streams with replace-on-encode (GBK/cp1252) safety."""
    import contextlib

    if sys.stdout is None:
        sys.stdout = _NullIO()  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _NullIO()  # type: ignore[assignment]
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):  # non-reconfigurable streams
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]


def _probe_existing(timeout_s: float = 0.6) -> str | None:
    """Base URL of an already-running CiteLens console, if any.

    Validated against /health's payload (not just an open port) so an
    unrelated app on 8000 doesn't hijack the browser tab.
    """
    import json as _json
    import urllib.request

    for port in range(8000, 8010):
        base = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(base + "/health", timeout=timeout_s) as r:
                info = _json.loads(r.read().decode("utf-8", "replace"))
            if info.get("status") == "ok" and "llm_model" in info:
                return base
        except Exception:  # noqa: BLE001 - not ours / not there
            continue
    return None


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


def _machine_home() -> Path:
    """The machine-wide CiteLens data home (last-resort default)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "CiteLens"


def _resolve_workdir(app_dir: Path) -> Path:
    """Where this launch keeps its data — the anti-"records vanished" rules.

    1. CITELENS_WORKDIR in the exe-folder .env explicitly redirects (the
       settings UI writes it); the .env migrates into the workdir as before.
    2. An exe-folder .env without a redirect = portable-folder mode: data
       and config live with THIS copy (existing setups keep working).
    3. Otherwise: a freshly downloaded exe must NOT present an empty
       workspace — reattach to the last workdir any launch used (pointer
       under the machine home), so the history follows the app, not the
       folder the exe happened to be downloaded into.
    4. First ever launch on the machine: the machine home itself.
    """
    from citens.api.envstore import read_env_value

    redirect = read_env_value(app_dir / ".env", "CITELENS_WORKDIR")
    if redirect:
        wd = Path(redirect)
        if not wd.is_absolute():
            wd = app_dir / wd
        wd.mkdir(parents=True, exist_ok=True)
        pointer = app_dir / ".env"
        target = wd / ".env"
        if pointer.is_file() and not target.is_file():
            keep = [
                line for line in pointer.read_text(encoding="utf-8").splitlines()
                if not line.strip().startswith("CITELENS_WORKDIR")
            ]
            target.write_text("\n".join(keep) + "\n", encoding="utf-8")
        _remember_workdir(wd)
        return wd

    if (app_dir / ".env").is_file():  # portable-folder mode
        _remember_workdir(app_dir)
        return app_dir

    home = _machine_home()
    last = home / "workdir.txt"
    if last.is_file():
        prev = Path(last.read_text(encoding="utf-8").strip())
        if prev.is_dir():
            return prev
    home.mkdir(parents=True, exist_ok=True)
    _remember_workdir(home)
    return home


def _remember_workdir(wd: Path) -> None:
    """Leave the pointer bare exe copies will reattach to next launch."""
    import contextlib

    with contextlib.suppress(Exception):
        home = _machine_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "workdir.txt").write_text(str(wd.resolve()), encoding="utf-8")


def main() -> None:
    _console_safe()
    # portable mode BEFORE any citens import (settings read .env from cwd).
    app_dir = _app_dir()
    os.chdir(_resolve_workdir(app_dir))

    if "--import-check" in sys.argv:  # exe smoke test / support tooling
        print(_import_selfcheck())
        return

    # single instance: double-clicking again must reuse the running console,
    # not spawn a second (terminal-less) server nobody can see or close
    existing = _probe_existing()
    if existing is not None:
        webbrowser.open(existing)
        return

    import uvicorn

    from citens.api.app import app
    from citens.api.envstore import read_env_value

    first_run = not read_env_value(Path.cwd() / ".env", "LLM_API_KEY")
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    url = base + ("/?setup=1" if first_run else "")
    print(f"  CiteLens console: {url}  (workdir: {Path.cwd()})")

    def _open_when_ready(timeout_s: int = 300) -> None:
        # a frozen exe's first launch spends 30-90s in self-extraction +
        # antivirus scanning BEFORE the server binds — opening the browser
        # early showed ERR_CONNECTION_REFUSED. Gate on /health instead.
        import urllib.request

        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        print(f"  ready ({time.monotonic() - t0:.0f}s), opening {url}")
                        webbrowser.open(url)
                        return
            except Exception:  # noqa: BLE001 - not ready yet
                pass
            time.sleep(1.0)

    threading.Thread(target=_open_when_ready, daemon=True).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    except BaseException:  # noqa: BLE001 - windowed builds show nothing; log it
        import contextlib

        err = app_dir / "error.log"
        with contextlib.suppress(Exception):
            err.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
