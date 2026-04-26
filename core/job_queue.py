from __future__ import annotations

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

