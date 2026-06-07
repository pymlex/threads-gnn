#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEED="${1:-42}"

python scripts/compare.py --seed "${SEED}"
python scripts/plot_curves.py --seed "${SEED}"
python scripts/plot_diagnostics.py --config configs/default.yaml --seed "${SEED}"

git add -f \
  runs/training_curves.png \
  runs/test_logit_histograms.png \
  runs/test_roc_curves.png \
  runs/architecture_comparison.csv \
  runs/selected_model.json \
  model_card.md

for architecture in gin pna gat; do
  run_dir="runs/${architecture}_seed${SEED}"
  git add -f \
    "${run_dir}/config.json" \
    "${run_dir}/epoch_metrics.csv" \
    "${run_dir}/final_metrics.json" \
    "${run_dir}/test_confusion_matrix.png" \
    "${run_dir}/test_logit_histogram.png" \
    "${run_dir}/test_roc_curve.png" \
    "${run_dir}/test_confusion_counts.csv" \
    "${run_dir}/test_confusion_normalised.csv" \
    "${run_dir}/test_classification_report.txt" \
    "${run_dir}/test_predictions.csv"
done

git commit -m "$(cat <<'EOF'
Add experiment metrics, training curves, and confusion matrices.

EOF
)"

git push origin main

echo "Results pushed to GitHub."
