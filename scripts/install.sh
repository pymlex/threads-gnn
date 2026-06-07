#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip install -q torch-geometric
python scripts/install_pyg.py
pip install -q -e .

if [ ! -f .env ]; then
  cp .env.example .env
fi

if ! command -v gh >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq gh
fi

if ! gh auth status >/dev/null 2>&1; then
  gh auth login --web --git-protocol https
fi

python -c "from utils.pyg_check import require_torch_scatter; require_torch_scatter(); print('torch-scatter OK')"
