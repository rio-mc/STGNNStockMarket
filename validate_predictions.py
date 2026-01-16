#!/usr/bin/env python3
"""
validate_predictions.py

Validate directional predictions in a results CSV by comparing entry vs exit
daily closes from Yahoo Finance.

Rules:
- No pandas_market_calendars dependency.
- Entry session = last available trading day on/before `features_end_ts` (interpreted in US/Eastern).
- Exit session  = next `horizon_days` trading sessions after entry (using Yahoo daily bars index).
- Correct if:
    Upwards   -> exit_close > entry_close
    Downwards -> exit_close < entry_close
  (Flat is treated as incorrect by default, but logged as actual_direction="Flat".)

Input CSV must include:
  - ticker
  - direction         ("Upwards" / "Downwards")
  - horizon           ("1d", "2d", ...)
  - features_end_ts   ISO timestamp (should ideally include timezone; if not, assumed UTC)

Output CSV = input columns +:
  - check_run_at_utc
  - features_end_ts_et
  - entry_session_date
  - exit_session_date
  - entry_close
  - exit_close
  - actual_direction
  - is_correct
  - abs_move
  - pct_move
  - validation_error
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Tuple

import pandas as pd


ET_TZ = "America/New_York"


def parse_horizon_days(horizon_str: str) -> int:
    s = str(horizon_str).strip().lower()
    if s.endswith("d"):
        n = int(s[:-1])
        if n < 1:
            raise ValueError("Horizon must be >= 1 day")
        return n
    raise ValueError(f"Unsupported horizon format: {horizon_str!r} (expected like '1d')")


def to_eastern(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Convert timestamp to US/Eastern. If tz-naive, assume UTC (explicit and consistent).
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ET_TZ)


def normalise_daily_history(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise yfinance output to a standard daily OHLCV DataFrame with tz-naive DateTimeIndex.
    """
    if hist is None or hist.empty:
        return pd.DataFrame()

    hist = hist.copy()

    # Handle MultiIndex columns in some yfinance outputs
    if isinstance(hist.columns, pd.MultiIndex):
        # Prefer (field, ticker) layout and select the first ticker slice
        tickers = list({c[1] for c in hist.columns})
        t0 = tickers[0] if tickers else None
        if t0 is not None:
            hist = hist.xs(t0, axis=1, level=1, drop_level=True)

    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    hist = hist.sort_index()
    return hist


def infer_entry_exit_from_daily(
    history: pd.DataFrame,
    features_end_ts_et: pd.Timestamp,
    n_days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Use Yahoo daily bars index as the trading-session calendar.
    Entry: last session on/before features_end_ts_et.date()
    Exit : n_days sessions after entry
    """
    if history.empty:
        raise ValueError("Empty daily history")

    sessions = history.index  # tz-naive daily dates
    target_day = pd.Timestamp(features_end_ts_et.date())  # tz-naive midnight

    prior = sessions[sessions <= target_day]
    if len(prior) == 0:
        raise ValueError("No session on/before features_end_ts in downloaded history")

    entry_date = prior[-1]
    entry_loc = sessions.get_loc(entry_date)
    exit_loc = entry_loc + int(n_days)

    if exit_loc >= len(sessions):
        raise ValueError("Not enough future sessions in downloaded history to validate")

    exit_date = sessions[exit_loc]
    return entry_date, exit_date


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input results CSV")
    ap.add_argument("--out_csv", required=True, help="Output validated CSV")
    ap.add_argument("--buffer_before_days", type=int, default=10)
    ap.add_argument("--buffer_after_days", type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    required = {"ticker", "direction", "horizon", "features_end_ts"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV missing required columns: {sorted(missing)}")

    check_run_at_utc = datetime.now(timezone.utc).isoformat()

    import yfinance as yf  # unofficial Yahoo Finance wrapper

    out_rows = []

    for _, r in df.iterrows():
        ticker = str(r["ticker"]).strip()
        pred_dir = str(r["direction"]).strip()
        horizon_days = parse_horizon_days(r["horizon"])

        # Convert features_end_ts to US/Eastern before choosing the session date
        features_end_ts_raw = pd.Timestamp(r["features_end_ts"])
        features_end_ts_et = to_eastern(features_end_ts_raw)

        row = dict(r)
        row.update(
            {
                "check_run_at_utc": check_run_at_utc,
                "features_end_ts_et": features_end_ts_et.isoformat(),
                "entry_session_date": None,
                "exit_session_date": None,
                "entry_close": None,
                "exit_close": None,
                "actual_direction": None,
                "is_correct": None,
                "abs_move": None,
                "pct_move": None,
                "validation_error": None,
            }
        )

        try:
            start = (features_end_ts_et - pd.Timedelta(days=int(args.buffer_before_days))).date().isoformat()
            end = (features_end_ts_et + pd.Timedelta(days=int(args.buffer_after_days))).date().isoformat()

            hist_raw = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            hist = normalise_daily_history(hist_raw)
            if hist.empty:
                raise ValueError("No Yahoo Finance daily data returned")
            if "Close" not in hist.columns:
                raise ValueError("Downloaded history has no 'Close' column")

            entry_date, exit_date = infer_entry_exit_from_daily(hist, features_end_ts_et, horizon_days)

            entry_close = float(hist.loc[entry_date, "Close"])
            exit_close = float(hist.loc[exit_date, "Close"])

            abs_move = exit_close - entry_close
            pct_move = (abs_move / entry_close) * 100.0 if entry_close != 0 else None

            if abs_move > 0:
                actual_dir = "Upwards"
            elif abs_move < 0:
                actual_dir = "Downwards"
            else:
                actual_dir = "Flat"

            if pred_dir == "Upwards":
                is_correct = abs_move > 0
            elif pred_dir == "Downwards":
                is_correct = abs_move < 0
            else:
                is_correct = None

            row["entry_session_date"] = entry_date.date().isoformat()
            row["exit_session_date"] = exit_date.date().isoformat()
            row["entry_close"] = entry_close
            row["exit_close"] = exit_close
            row["actual_direction"] = actual_dir
            row["is_correct"] = bool(is_correct) if is_correct is not None else None
            row["abs_move"] = abs_move
            row["pct_move"] = pct_move

        except Exception as e:
            row["validation_error"] = str(e)

        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
