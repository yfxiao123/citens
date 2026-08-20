"""Desktop entry point for the single-exe build (PyInstaller).

Doubles as the ``citens-desktop`` console script in normal installs. The
exe is meant to feel like a desktop app:

    double-click CiteLens.exe
      -> first run: a short console wizard writes .env next to the exe
      -> every run: local web console starts and the browser opens

Portability rule: EVERYTHING lives next to the exe — .env, .cache, papers/,
runs/, data/. Copy the folder to another machine and it just works; delete
it and nothing is left behind.

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

_PROVIDER_PRESETS = {
    "1": ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    "2": ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    "3": ("Ollama (local)", "http://localhost:11434/v1", "qwen2.5:7b"),
}


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _app_dir() -> Path:
    """The portable app folder: the exe's directory (cwd in dev mode)."""
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def _first_run_wizard(env_path: Path) -> None:
    print()
    print("=" * 62)
    print("  CiteLens 首次运行配置 / first-run setup")
    print("  配置会保存到 exe 旁边的 .env（可随时用记事本修改）")
    print("=" * 62)
    print()
    print("选择 LLM 服务商 / choose your LLM provider:")
    for k, (name, _, _) in _PROVIDER_PRESETS.items():
        print(f"  {k}. {name}")
    print("  4. 其他 OpenAI 兼容服务 (OpenRouter / vLLM / Groq ...)")
    choice = input("选择 (1) > ").strip() or "1"

    if choice in _PROVIDER_PRESETS:
        name, base, default_model = _PROVIDER_PRESETS[choice]
    else:
        name, base, default_model = "custom", "", ""
        base = input("API Base URL (如 https://openrouter.ai/api/v1): ").strip()
        while not base:
            base = input("API Base URL 不能为空: ").strip()

    key = input(f"{name} API Key: ").strip()
    while not key:
        key = input("API Key 不能为空 / key is required: ").strip()

    model = input(f"模型名 (默认 {default_model}) > ").strip() or default_model

    env_path.write_text(
        f"# written by the CiteLens desktop first-run wizard\n"
        f"LLM_PROVIDER=openai\n"
        f"LLM_API_BASE={base}\n"
        f"LLM_API_KEY={key}\n"
        f"LLM_MODEL={model}\n",
        encoding="utf-8",
    )
    print()
    print(f"已保存 {env_path} — 正在启动控制台…")


def _needs_wizard(env_path: Path) -> bool:
    if not env_path.is_file():
        return True
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("LLM_API_KEY") and len(s.split("=", 1)[-1].strip()) > 8:
            return False
    return True


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
    # portable mode BEFORE any citens import (settings read .env from cwd)
    app_dir = _app_dir()
    os.chdir(app_dir)

    if "--import-check" in sys.argv:  # exe smoke test / support tooling
        print(_import_selfcheck())
        return

    env_path = app_dir / ".env"
    if _needs_wizard(env_path):
        _first_run_wizard(env_path)

    import uvicorn

    from citens.api.app import app

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print()
    print(f"  CiteLens 控制台运行中: {url}   (Ctrl+C 退出)")
    print(f"  工作目录: {app_dir}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\n  bye")


if __name__ == "__main__":
    main()
