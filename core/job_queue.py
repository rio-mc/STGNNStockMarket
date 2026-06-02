from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import List, Optional


@dataclass(frozen=True)
class QueueJob:
    job_id: str
    created_at: str
    prediction_window: str
    ticker: str
    model: str
    seed: int
    graph_model: str = "gcn"
    k: int = 3
    graph_mode: str = "knn_mst"
    graph_embed: str = "pca"
    graph_ablation: str = "none"
    ablate_feature: str = "none"
    seq_len: int = 10
    batch_size: int = 256
    lstm_epochs: int = 200
    stgnn_epochs: int = 200


class JobQueueController:
    def __init__(self) -> None:
        self._jobs: List[QueueJob] = []
        self._lock = Lock()

    def enqueue(self, job: QueueJob) -> None:
        with self._lock:
            self._jobs.append(job)

    def enqueue_many(self, jobs: List[QueueJob]) -> None:
        with self._lock:
            self._jobs.extend(jobs)

    def pop_next(self) -> Optional[QueueJob]:
        with self._lock:
            if not self._jobs:
                return None
            return self._jobs.pop(0)

    def remove_at(self, index: int) -> Optional[QueueJob]:
        with self._lock:
            if index < 0 or index >= len(self._jobs):
                return None
            return self._jobs.pop(index)

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def snapshot(self) -> List[QueueJob]:
        with self._lock:
            return list(self._jobs)

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    @staticmethod
    def make_job_id(prefix: str = "JOB") -> str:
        return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"


def parse_seed_spec(seed_spec: str) -> List[int]:
    """
    Supports:
    - 42
    - 42,43,44
    - 42-46
    - 42,50-52
    """
    raw = str(seed_spec or "").strip()
    if not raw:
        return [42]

    values: List[int] = []
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    for part in parts:
        if "-" in part:
            start_s, end_s = [x.strip() for x in part.split("-", 1)]
            start_i = int(start_s)
            end_i = int(end_s)
            if end_i < start_i:
                raise ValueError(f"Invalid seed range: {part}")
            values.extend(range(start_i, end_i + 1))
        else:
            values.append(int(part))

    seen = set()
    deduped: List[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _normalise_universe(available_tickers: List[str]) -> List[str]:
    return _dedupe_keep_order(
        [str(t).strip().upper() for t in available_tickers if str(t).strip()]
    )


def _parse_explicit_ticker_list(raw: str, available_tickers: List[str]) -> List[str]:
    universe_set = set(available_tickers)
    requested = _dedupe_keep_order(
        [part.strip().upper() for part in raw.split(",") if part.strip()]
    )

    missing = [t for t in requested if t not in universe_set]
    if missing:
        raise ValueError(
            f"Unknown ticker(s): {', '.join(missing)}. "
            f"Available universe count={len(available_tickers)}"
        )

    return requested


def _parse_random_spec(
    raw: str,
    available_tickers: List[str],
    rng_seed: Optional[int],
) -> List[str]:
    match = re.fullmatch(r"random\s*:\s*(\d+)", raw, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Random selector must look like 'random:10'.")

    count = int(match.group(1))
    if count < 1:
        raise ValueError("Random selector count must be >= 1.")
    if count > len(available_tickers):
        raise ValueError(
            f"Random selector requested {count} tickers but only "
            f"{len(available_tickers)} available."
        )

    rng = random.Random(rng_seed if rng_seed is not None else None)
    sample = rng.sample(list(available_tickers), count)
    return sorted(sample)


def _parse_alpha_spec(raw: str, available_tickers: List[str]) -> List[str]:
    """
    Supports:
    - alpha:a-g
    - alpha:m
    - alpha:ms-pm

    Prefix comparison is lexicographic against the ticker symbol.
    """
    match = re.fullmatch(
        r"alpha\s*:\s*([a-z]+)(?:\s*-\s*([a-z]+))?",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Alpha selector must look like 'alpha:a-g' or 'alpha:m'.")

    start = match.group(1).upper()
    end = (match.group(2) or start).upper()

    if end < start:
        raise ValueError(f"Invalid alpha range: {start}-{end}")

    prefix_len = max(len(start), len(end))
    selected = [
        ticker
        for ticker in sorted(available_tickers)
        if start <= ticker[:prefix_len] <= end
    ]

    if not selected:
        raise ValueError(f"No tickers matched alpha selector '{raw}'.")
    return selected


def parse_ticker_spec(
    ticker_spec: str,
    available_tickers: List[str],
    rng_seed: Optional[int] = None,
) -> List[str]:
    """
    Supports:
    - AAPL
    - AAPL,MSFT,NVDA
    - all
    - *
    - random:10
    - alpha:a-g
    - alpha:m
    - alpha:ms-pm
    """
    universe = sorted(_normalise_universe(available_tickers))
    raw = str(ticker_spec or "").strip()

    if not universe:
        raise ValueError("No available tickers in universe.")
    if not raw:
        raise ValueError("Please enter a ticker selection.")

    lowered = raw.lower()

    if lowered in {"all", "*"}:
        return universe

    if lowered.startswith("random:"):
        return _parse_random_spec(raw, universe, rng_seed=rng_seed)

    if lowered.startswith("alpha:"):
        return _parse_alpha_spec(raw, universe)

    return _parse_explicit_ticker_list(raw, universe)
