"""Small helper for persisting graph statistics under results/graph_logging/."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np


def _clean_for_json(stats: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in stats.items():
        if isinstance(value, float) and np.isnan(value):
            cleaned[key] = None
        elif isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def save_graph_stats(stats: Dict[str, Any], output_path: str = "results/graph_logging/graph_stats.json") -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_clean_for_json(stats), f, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy graph stats JSON into the results graph-logging folder.")
    parser.add_argument("--input", default="results/graph_stats.json", help="Existing graph_stats.json to copy from.")
    parser.add_argument("--output", default="results/graph_logging/graph_stats.json", help="Destination JSON path.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        stats = json.load(f)
    path = save_graph_stats(stats, args.output)
    print(f"Saved graph stats to {path}")


if __name__ == "__main__":
    main()
