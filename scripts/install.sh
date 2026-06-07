#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -q torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install -q -e .

if [ ! -f .env ]; then
  cp .env.example .env
fi

if ! command -v gh >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq gh
fi

gh auth login --web --git-protocol https
