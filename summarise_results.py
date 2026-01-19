#!/usr/bin/env python3
"""
Summarise results/benchmark_run.csv into labelled suite averages.

Outputs (in ./summaries next to this script):
  - suite_averages.csv
  - suite_averages_pretty.csv
  - suite_averages.md
  - per_ticker.csv
  - completeness.csv
  - overall_by_model.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

# -------------------------
# Settings (no CLI)
# -------------------------
DEFAULT_METRICS = [
    # threshold-aligned (your "best threshold" operating point)
    "accuracy", "macro_f1", "roc_auc", "ap",
    # fixed 0.5 operating point
    "accuracy_05", "macro_f1_05",
    # efficiency
    "energy_Wh", "train_seconds",
]
DECIMALS = 4

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "results" / "benchmark_run.csv"
OUTDIR = SCRIPT_DIR / "summaries"


def build_suite_label(row: pd.Series) -> str:
    """Human-readable suite label from config columns."""
    bits = []

    ga = row.get("graph_ablation", "unknown")
    if ga == "identity":
        bits.append("graph=self-loops")
    elif ga == "none":
        bits.append("graph=relational")
    else:
        bits.append(f"graph={ga}")

    ablated = row.get("ablated", None)
    if pd.isna(ablated) or str(ablated) == "none":
        bits.append("descriptor=full")
    else:
        bits.append(f"descriptor={ablated}")

    k = row.get("max_k", None)
    if pd.isna(k):
        bits.append("k=?")
    else:
        bits.append(f"k={int(k)}")

    if "rewiring" in row.index and not pd.isna(row["rewiring"]):
        bits.append(f"rewire={'on' if bool(row['rewiring']) else 'off'}")

    if "seq_len" in row.index and not pd.isna(row["seq_len"]):
        bits.append(f"L={int(row['seq_len'])}")
    if "horizon" in row.index and not pd.isna(row["horizon"]):
        bits.append(f"H={row['horizon']}")

    return " | ".join(bits)


def mean_std_str(mean: float, std: float, decimals: int = 4) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from: {INPUT_CSV}")

    # Sanity checks
    required = {"model", "ticker", "seed", "graph_ablation", "ablated", "max_k"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Normalise rewiring if present
    if "rewiring" in df.columns:
        # handles bool/int/str-ish values robustly
        df["rewiring"] = df["rewiring"].apply(lambda x: bool(int(x)) if str(x).isdigit() else bool(x))

    # Build suite label
    df["suite"] = df.apply(build_suite_label, axis=1)

    # Keep only metrics that actually exist in the CSV
    metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    if not metrics:
        raise SystemExit(f"None of DEFAULT_METRICS were found in CSV. DEFAULT_METRICS={DEFAULT_METRICS}")

    # Expected rows per (model, suite): tickers * seeds
    n_tickers = df["ticker"].nunique()
    n_seeds = df["seed"].nunique()
    expected = n_tickers * n_seeds

    # 1) Suite-level averages across all tickers and seeds
    group_cols = ["model", "suite"]
    g = df.groupby(group_cols, dropna=False)

    suite_stats = g[metrics].agg(["mean", "std", "count"]).reset_index()

    # Flatten multiindex columns
    suite_stats.columns = [
        f"{a}_{b}" if b else a
        for (a, b) in [(c if isinstance(c, tuple) else (c, "")) for c in suite_stats.columns]
    ]

    # Completeness (use count from first available metric)
    count_col = f"{metrics[0]}_count"
    suite_stats["expected_rows"] = expected
    suite_stats["missing_rows"] = suite_stats["expected_rows"] - suite_stats[count_col].astype(int)
    suite_stats["complete"] = suite_stats["missing_rows"].eq(0)

    suite_stats.to_csv(OUTDIR / "suite_averages.csv", index=False)

    # Pretty markdown summary (mean ± std)
    pretty_rows = []
    for _, r in suite_stats.iterrows():
        row = {"model": r["model"], "suite": r["suite"], "n": int(r[count_col])}
        for m in metrics:
            row[m] = mean_std_str(r[f"{m}_mean"], r[f"{m}_std"], DECIMALS)
        pretty_rows.append(row)

    pretty = pd.DataFrame(pretty_rows)
    pretty.to_csv(OUTDIR / "suite_averages_pretty.csv", index=False)
    pretty.to_markdown(OUTDIR / "suite_averages.md", index=False)

    # 2) Per-ticker summaries (average over seeds)
    per_ticker = df.groupby(["model", "suite", "ticker"], dropna=False)[metrics].mean().reset_index()
    per_ticker.to_csv(OUTDIR / "per_ticker.csv", index=False)

    # 3) Completeness report by suite/model
    completeness = df.groupby(["model", "suite"], dropna=False).size().reset_index(name="rows")
    completeness["expected_rows"] = expected
    completeness["missing_rows"] = completeness["expected_rows"] - completeness["rows"]
    completeness["complete"] = completeness["missing_rows"].eq(0)
    completeness.to_csv(OUTDIR / "completeness.csv", index=False)

    # 4) Overall model-level summary (collapsing suites)
    overall = df.groupby(["model"], dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    overall.columns = ["model"] + [f"{m}_{stat}" for m, stat in overall.columns.tolist()[1:]]
    overall.to_csv(OUTDIR / "overall_by_model.csv", index=False)

    print(f"Wrote summaries to: {OUTDIR}")


if __name__ == "__main__":
    main()
