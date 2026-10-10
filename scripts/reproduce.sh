#!/usr/bin/env bash
# Reproduces every result: data, pre-flight, training, evaluation, report.
#
#   scripts/reproduce.sh                 # full run (hours)
#   STEPS=300 scripts/reproduce.sh       # quick end-to-end check
#
# Each stage writes its artifact under runs/ and is skipped if it exists, so
# an interrupted run resumes where it stopped (training resumes from its last
# checkpoint). The dry run gates training: if it is not clean, nothing trains.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
STEPS="${STEPS:-12000}"
OUT="${OUT:-ckpts/r-small}"
CONTROL="${CONTROL:-ckpts/r-nostate}"
export HF_HUB_DISABLE_TELEMETRY=1

[ -f data/train.bin ] || "$PY" scripts/prepare_data.py --out data

"$PY" scripts/dry_run.py --preset small --out-json runs/dry-run.json

"$PY" scripts/train_r.py --preset small --out-dir "$OUT" --resume \
    --set "train.steps=$STEPS"
cp "$OUT/record.json" runs/final-record.json

[ -f runs/eval.json ] || "$PY" scripts/eval.py \
    --recurrent-ckpt "$OUT/final.pt" --out-json runs/eval.json

[ -f runs/compress.json ] || "$PY" scripts/compress.py \
    --ckpt "$OUT/final.pt" --tokens 4096 --out-json runs/compress.json

if [ -f "$CONTROL/final.pt" ]; then
    "$PY" scripts/probe_state.py --ckpt "$OUT/final.pt" \
        --control "$CONTROL/final.pt" --out-json runs/probe-state.json
fi

"$PY" scripts/report.py --out paper/RESULTS.md
