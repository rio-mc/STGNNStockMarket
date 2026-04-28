from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Iterable
from uuid import uuid4


@dataclass
class RunRecord:
    # Required core run fields
    run_id: str
    job_id: Optional[str]
    status: str
    timestamp_start: str
    timestamp_end: str
    duration_sec: float

    # Required experiment context
    ticker: str
    prediction_window: str
    model: str
    seed: int
    graph_model: str

    universe_id: str
    interval: str
    k: int
    graph_mode: str
    graph_embed: str
    ablate_feature: str
    threshold_policy: str

    # Optional queue grouping
    queue_run_id: Optional[str] = None
    queue_group: Optional[str] = None

    # Optional result payload
    direction: Optional[str] = None
    confidence: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


class ExperimentStore:
    def __init__(self, root_dir: str | Path = "./results") -> None:
        self.root_dir = Path(root_dir)
        self.runs_dir = self.root_dir / "runs"
        self.histories_dir = self.runs_dir / "histories"
        self.queue_runs_dir = self.runs_dir / "queue_runs"

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.histories_dir.mkdir(parents=True, exist_ok=True)
        self.queue_runs_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.runs_dir / "experiments.jsonl"
        self.csv_path = self.runs_dir / "experiments.csv"

    @staticmethod
    def make_run_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"RUN-{ts}-{uuid4().hex[:8]}"

    @staticmethod
    def make_queue_run_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"QUEUE-{ts}-{uuid4().hex[:8]}"

    @staticmethod
    def model_group(model: str, graph_model: str = "") -> str:
        model_key = str(model or "").strip().lower()
        graph_key = str(graph_model or "").strip().lower()

        if model_key in {"lstm", "gru"}:
            return "recurrent"

        if model_key == "stgnn":
            return "stgnn"

        if model_key in {"gcn", "gat", "graphsage", "nnconv"}:
            return "graph"

        # STGNN can carry a graph backend in graph_model; keep it separately.
        if graph_key in {"gcn", "gat", "graphsage", "nnconv"}:
            return "graph"

        return "other"

    def queue_run_dir(self, queue_run_id: str) -> Path:
        path = self.queue_runs_dir / str(queue_run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def queue_job_dir(
        self,
        *,
        queue_run_id: str,
        model: str,
        graph_model: str = "",
        job_id: str,
    ) -> Path:
        group = self.model_group(model, graph_model)
        model_name = str(model or "unknown").strip().lower() or "unknown"
        job_name = str(job_id or "job").strip() or "job"

        path = self.queue_run_dir(queue_run_id) / group / model_name / job_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_queue_manifest(
        self,
        *,
        queue_run_id: str,
        status: str,
        timestamp_start: str,
        timestamp_end: str | None = None,
        jobs: Iterable[Any] | None = None,
        completed: int = 0,
        failed: int = 0,
        cancelled: int = 0,
        extras: Dict[str, Any] | None = None,
    ) -> str:
        run_dir = self.queue_run_dir(queue_run_id)

        payload = {
            "queue_run_id": queue_run_id,
            "status": status,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "completed": int(completed),
            "failed": int(failed),
            "cancelled": int(cancelled),
            "jobs": [self._normalise(j) for j in (jobs or [])],
            "extras": self._normalise(extras or {}),
        }

        manifest_path = run_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._normalise(payload), f, ensure_ascii=False, indent=2)

        return str(manifest_path)

    def save_job_payload(
        self,
        *,
        queue_run_id: str,
        job_id: str,
        model: str,
        graph_model: str = "",
        filename: str,
        payload: Any,
    ) -> str:
        job_dir = self.queue_job_dir(
            queue_run_id=queue_run_id,
            model=model,
            graph_model=graph_model,
            job_id=job_id,
        )
        path = job_dir / filename

        with path.open("w", encoding="utf-8") as f:
            json.dump(self._normalise(payload), f, ensure_ascii=False, indent=2)

        return str(path)

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def append_run(self, record: RunRecord) -> None:
        payload = self._normalise(asdict(record))

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self._append_csv_summary(payload)

    def save_history(
        self,
        run_id: str,
        hist_train: Optional[list[Any]] = None,
        hist_val: Optional[list[Any]] = None,
        *,
        queue_run_id: Optional[str] = None,
        job_id: Optional[str] = None,
        model: str = "",
        graph_model: str = "",
    ) -> Optional[str]:
        if not hist_train and not hist_val:
            return None

        if queue_run_id and job_id:
            path = self.queue_job_dir(
                queue_run_id=queue_run_id,
                model=model,
                graph_model=graph_model,
                job_id=job_id,
            ) / "history.json"
        else:
            path = self.histories_dir / f"{run_id}.json"
        payload = {
            "run_id": run_id,
            "hist_train": self._normalise(hist_train or []),
            "hist_val": self._normalise(hist_val or []),
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return str(path)

    def write_queue_seed_summaries(
        self,
        *,
        queue_run_id: str,
        job_summaries: list[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Write research-suite summaries for a completed queue run.

        Outputs:
          queue_runs/<queue_run_id>/seed_results.csv
          queue_runs/<queue_run_id>/summary_by_model.csv
          queue_runs/<queue_run_id>/<group>/<model>/seed_results.csv
          queue_runs/<queue_run_id>/<group>/<model>/summary.json

        The seed_results files are one row per executed job/seed.
        The summary files aggregate numeric metrics across seeds.
        """
        run_dir = self.queue_run_dir(queue_run_id)
        rows = [self._flatten_queue_job_summary(r) for r in (job_summaries or [])]

        paths: Dict[str, str] = {}

        if not rows:
            return paths

        all_seed_path = run_dir / "seed_results.csv"
        self._write_dict_rows_csv(all_seed_path, rows)
        paths["seed_results_csv"] = str(all_seed_path)

        aggregate_rows = self._aggregate_queue_rows(rows)
        summary_path = run_dir / "summary_by_model.csv"
        self._write_dict_rows_csv(summary_path, aggregate_rows)
        paths["summary_by_model_csv"] = str(summary_path)

        grouped: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
        for row in rows:
            key = (
                str(row.get("queue_group") or "other"),
                str(row.get("model") or "unknown"),
            )
            grouped.setdefault(key, []).append(row)

        for (group, model), group_rows in grouped.items():
            group_dir = run_dir / group / model
            group_dir.mkdir(parents=True, exist_ok=True)

            group_seed_path = group_dir / "seed_results.csv"
            self._write_dict_rows_csv(group_seed_path, group_rows)

            group_summary_rows = self._aggregate_queue_rows(group_rows)
            group_summary = group_summary_rows[0] if group_summary_rows else {}

            group_summary_path = group_dir / "summary.json"
            with group_summary_path.open("w", encoding="utf-8") as f:
                json.dump(self._normalise(group_summary), f, ensure_ascii=False, indent=2)

            paths[f"{group}_{model}_seed_results_csv"] = str(group_seed_path)
            paths[f"{group}_{model}_summary_json"] = str(group_summary_path)

        return paths

    def _flatten_queue_job_summary(self, row: Dict[str, Any]) -> Dict[str, Any]:
        metrics = row.get("metrics") or {}

        flat: Dict[str, Any] = {
            "queue_run_id": row.get("queue_run_id"),
            "queue_group": row.get("queue_group"),
            "run_id": row.get("run_id"),
            "job_id": row.get("job_id"),
            "status": row.get("status"),
            "ticker": row.get("ticker"),
            "prediction_window": row.get("prediction_window"),
            "model": row.get("model"),
            "seed": row.get("seed"),
            "graph_model": row.get("graph_model"),
            "direction": row.get("direction"),
            "confidence": row.get("confidence"),
            "error_message": row.get("error_message"),
        }

        for key, value in metrics.items():
            flat[f"metric_{key}"] = self._normalise(value)

        return flat

    def _write_dict_rows_csv(self, path: Path, rows: list[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            return

        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _aggregate_queue_rows(self, rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        Aggregate numeric metrics across seeds per queue_group/model/graph/window.
        """
        grouped: Dict[tuple, list[Dict[str, Any]]] = {}

        for row in rows:
            key = (
                row.get("queue_group"),
                row.get("model"),
                row.get("graph_model"),
                row.get("prediction_window"),
            )
            grouped.setdefault(key, []).append(row)

        out: list[Dict[str, Any]] = []

        for (queue_group, model, graph_model, prediction_window), group_rows in grouped.items():
            summary: Dict[str, Any] = {
                "queue_group": queue_group,
                "model": model,
                "graph_model": graph_model,
                "prediction_window": prediction_window,
                "n_jobs": len(group_rows),
                "n_success": sum(1 for r in group_rows if r.get("status") == "success"),
                "n_failed": sum(1 for r in group_rows if r.get("status") == "failed"),
                "n_cancelled": sum(1 for r in group_rows if r.get("status") == "cancelled"),
                "seeds": ",".join(str(r.get("seed")) for r in group_rows if r.get("seed") not in ("", None)),
                "tickers": ",".join(sorted({str(r.get("ticker")) for r in group_rows if r.get("ticker")})),
            }

            numeric_keys = sorted(
                {
                    key
                    for row in group_rows
                    for key, value in row.items()
                    if key.startswith("metric_") or key in {"confidence"}
                    if self._coerce_float(value) is not None
                }
            )

            for key in numeric_keys:
                values = [self._coerce_float(r.get(key)) for r in group_rows]
                values = [v for v in values if v is not None]

                if not values:
                    continue

                mean_v = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((v - mean_v) ** 2 for v in values) / (len(values) - 1)
                    std_v = variance ** 0.5
                else:
                    std_v = 0.0

                summary[f"{key}_mean"] = mean_v
                summary[f"{key}_std"] = std_v
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_n"] = len(values)

            out.append(summary)

        return out

    def _coerce_float(self, value: Any) -> Optional[float]:
        try:
            if value is None or isinstance(value, bool):
                return None
            value = float(value)
            if value != value:
                return None
            if value in (float("inf"), float("-inf")):
                return None
            return value
        except Exception:
            return None

    def _append_csv_summary(self, payload: Dict[str, Any]) -> None:
        flat = self._flatten_for_csv(payload)

        if self.csv_path.exists():
            existing_header = self._read_csv_header()
            if existing_header is not None and existing_header != list(flat.keys()):
                flat = self._rewrite_csv_with_union_schema(flat)
                return

        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(flat)

    def _read_csv_header(self) -> Optional[list[str]]:
        if not self.csv_path.exists():
            return None

        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                return next(reader)
            except StopIteration:
                return None

    def _rewrite_csv_with_union_schema(self, new_row: Dict[str, Any]) -> Dict[str, Any]:
        old_rows: list[Dict[str, Any]] = []
        old_header = self._read_csv_header() or []

        if self.csv_path.exists():
            with self.csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                old_rows = list(reader)

        union_keys = list(dict.fromkeys(old_header + list(new_row.keys())))

        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=union_keys)
            writer.writeheader()

            for row in old_rows:
                writer.writerow({key: row.get(key, "") for key in union_keys})

            writer.writerow({key: new_row.get(key, "") for key in union_keys})

        return new_row

    def _flatten_for_csv(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        metrics = payload.get("metrics", {}) or {}
        extras = payload.get("extras", {}) or {}

        flat: Dict[str, Any] = {
            "run_id": payload.get("run_id"),
            "job_id": payload.get("job_id"),
            "queue_run_id": payload.get("queue_run_id"),
            "queue_group": payload.get("queue_group"),
            "status": payload.get("status"),
            "timestamp_start": payload.get("timestamp_start"),
            "timestamp_end": payload.get("timestamp_end"),
            "duration_sec": payload.get("duration_sec"),
            "ticker": payload.get("ticker"),
            "prediction_window": payload.get("prediction_window"),
            "model": payload.get("model"),
            "seed": payload.get("seed"),
            "graph_model": payload.get("graph_model"),
            "universe_id": payload.get("universe_id"),
            "interval": payload.get("interval"),
            "k": payload.get("k"),
            "graph_mode": payload.get("graph_mode"),
            "graph_embed": payload.get("graph_embed"),
            "ablate_feature": payload.get("ablate_feature"),
            "threshold_policy": payload.get("threshold_policy"),
            "direction": payload.get("direction"),
            "confidence": payload.get("confidence"),
        }

        for key, value in metrics.items():
            flat[f"metric_{key}"] = self._normalise(value)

        for key, value in extras.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                flat[f"extra_{key}"] = value
            else:
                flat[f"extra_{key}"] = json.dumps(self._normalise(value), ensure_ascii=False)

        return flat

    def _normalise(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._normalise(asdict(value))

        if isinstance(value, dict):
            return {str(k): self._normalise(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._normalise(v) for v in value]

        if isinstance(value, tuple):
            return [self._normalise(v) for v in value]

        try:
            import numpy as np  # type: ignore

            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass

        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)