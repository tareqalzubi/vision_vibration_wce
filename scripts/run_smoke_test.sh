#!/usr/bin/env bash
set -euo pipefail
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} python scripts/make_synthetic_smoke_data.py --out data/smoke --num-sequences 2 --frames-per-sequence 8 --image-size 64
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} PYTHONPATH=src python train.py --config configs/smoke.yaml
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} PYTHONPATH=src python eval.py --config configs/smoke.yaml --checkpoint outputs/smoke/checkpoints/best.pt
