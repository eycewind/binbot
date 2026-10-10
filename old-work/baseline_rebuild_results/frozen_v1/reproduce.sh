#!/usr/bin/env bash
set -euo pipefail
conda run -n ml-2 python -m unittest -v test_baseline_rebuild.py test_run_baseline_rebuild.py
conda run -n ml-2 python run_baseline_rebuild.py --db ../db/BTCUSDT_1m_52weeks.db --output-dir baseline_rebuild_results/reproduced_run
