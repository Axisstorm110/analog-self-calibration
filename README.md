# Analog Self-Calibration

Research code for our analog in-memory training project. Building on Wu et al. 2025 (analog-training) and FALCON.

## Setup
- Python 3.10+
- `pip install aihwkit` (1.1.0 works, ignore the 0.9.0 pin in the original repo)
- Do NOT use the original repo's requirements.txt, it's a broken conda export
- Run their scripts from repo root with `PYTHONPATH=.` or you hit a `utils` import error

## Structure
- `experiments/` — training runs and configs
- `drift_model/` — device response modeling
- `figures/` — plots
- `paper/` — writeup

## Status
Phase 2 baseline reproduced (Analog SGD plateau, MNIST FP 98.1% vs Analog 96.4%).
