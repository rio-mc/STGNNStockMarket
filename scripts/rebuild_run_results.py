"""Rebuild flat global and model-family CSVs from experiments.jsonl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_store import ExperimentStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args(argv)

    outcome = ExperimentStore(args.results_dir).rebuild_run_result_csvs()
    print(f"Rebuilt {outcome['records']} run rows.")
    for family, path in outcome["paths"].items():
        print(f"{family}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
