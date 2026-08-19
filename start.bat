@echo off
setlocal
cd /d "%~dp0"

rem ---- CiteLens one-click launcher (Windows) ------------------------------
rem  Double-click: pick a toolchain (uv preferred, else python/py) ->
rem  create .venv -> install deps -> open the browser console.
rem  Requirement: uv (https://docs.astral.sh/uv/) OR Python 3.10+ on PATH.
rem  NOTE: keep this file ASCII-only. cmd.exe parses .bat in the ANSI
rem  codepage; non-ASCII bytes corrupt the line structure (mojibake errors).

set PY=

where uv >nul 2>nul
if not errorlevel 1 goto have_uv

where python >nul 2>nul
if not errorlevel 1 ( set PY=python & goto have_py )

where py >nul 2>nul
if not errorlevel 1 ( set PY=py -3 & goto have_py )

echo [x] no python toolchain found.
echo     install uv      : https://docs.astral.sh/uv/  (fastest)
echo     or Python 3.10+ : https://www.python.org/downloads/
pause
exit /b 1

:have_uv
if not exist .venv (
  echo [1/3] creating .venv with uv ...
  uv venv .venv
)
if not exist ".venv\Scripts\citens.exe" (
  echo [2/3] installing dependencies with uv, first run takes a minute ...
  uv pip install -q -e ".[api]"
) else (
  echo [2/3] dependencies ready
)
goto run

:have_py
if not exist .venv (
  echo [1/3] creating .venv ...
  %PY% -m venv .venv
)
if not exist ".venv\Scripts\citens.exe" (
  echo [2/3] installing dependencies, first run takes 1-2 minutes ...
  ".venv\Scripts\python.exe" -m pip install -q -e ".[api]"
) else (
  echo [2/3] dependencies ready
)
goto run

:run
if not exist ".venv\Scripts\citens.exe" (
  echo [x] dependency install failed - see messages above, then re-run
  pause
  exit /b 1
)
if not exist .env (
  echo [3/3] .env missing - copied from template, please fill in LLM_API_KEY
  copy .env.example .env >nul
  notepad .env
)
echo starting CiteLens Console ...
".venv\Scripts\citens.exe" serve --open
pause
