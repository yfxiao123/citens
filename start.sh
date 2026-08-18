#!/usr/bin/env bash
# ---- CiteLens 一键启动 (macOS / Linux) -----------------------------------
#  ./start.sh  即可: 自动建 venv -> 装依赖 -> 打开浏览器控制台
#  前提: python3 (3.10+) 在 PATH 中
set -e
cd "$(dirname "$0")"

PY=python3
command -v $PY >/dev/null 2>&1 || { echo "[x] 未找到 python3 — 请先安装 Python 3.10+"; exit 1; }

if [ ! -d .venv ]; then
  echo "[1/3] 创建虚拟环境 .venv ..."
  $PY -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import citens, fastapi" >/dev/null 2>&1; then
  echo "[2/3] 安装依赖 (首次约 1-2 分钟) ..."
  python -m pip install -q -e ".[api,pdf]"
else
  echo "[2/3] 依赖已就绪"
fi

if [ ! -f .env ]; then
  echo "[3/3] 未找到 .env — 从模板复制，请编辑填入 LLM_API_KEY"
  cp .env.example .env
  ${EDITOR:-nano} .env
fi

echo "正在启动 CiteLens Console ..."
citens serve --open
