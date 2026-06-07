#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TORCH_TAG=$(python -c "import torch; v=torch.__version__.split('+')[0]; c=torch.version.cuda; print('torch-'+v+'+cu'+c.replace('.','') if c else 'torch-'+v+'+cpu')")
PYG_INDEX="https://data.pyg.org/whl/${TORCH_TAG}.html"

pip install -q torch-geometric
pip install -q pyg-lib torch-scatter torch-sparse -f "${PYG_INDEX}"
pip install -q -e .

if [ ! -f .env ]; then
  cp .env.example .env
fi

if ! command -v gh >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq gh
fi

gh auth login --web --git-protocol https
