"""Deprecated compatibility wrapper.

Use run_experiment.py for canonical headless runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment import main, run_experiment, save_results


if __name__ == "__main__":
    main()
