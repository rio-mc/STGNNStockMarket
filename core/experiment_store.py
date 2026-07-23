from __future__ import annotations

import csv
import json
import math
import time
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
    graph_ablation: str
    ablate_feature: str
    threshold_policy: str
    deterministic: bool

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
        self.run_results_csv_path = self.runs_dir / "run_results.csv"

    @staticmethod
    def _write_json_payload(path: Path, payload: Any) -> None:
        """Write JSON atomically, tolerating transient removal of a parent folder."""

        last_error: OSError | None = None
        for attempt in range(3):
            temporary_path = path.with_name(
                f".{path.name}.{uuid4().hex}.tmp"
            )
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with temporary_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                temporary_path.replace(path)
                return
            except (FileNotFoundError, PermissionError) as exc:
                last_error = exc
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1))

        assert last_error is not None
        raise last_error

    def _append_jsonl_payload(self, payload: Dict[str, Any]) -> None:
        """Append the canonical run record, recreating storage if it vanished."""

        last_error: OSError | None = None
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        for attempt in range(3):
            try:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
                return
            except (FileNotFoundError, PermissionError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1))

        assert last_error is not None
        raise last_error

    @staticmethod
    def make_run_id(
        *,
        model: str | None = None,
        ticker: str | None = None,
        seed: int | None = None,
        graph_backend: str | None = None,
    ) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

        parts = ["RUN", ts]

        if model:
            parts.append(str(model).strip().lower())

        if graph_backend:
            parts.append(str(graph_backend).strip().lower())

        if ticker:
            parts.append(str(ticker).strip().upper())

        if seed is not None:
            parts.append(f"seed{int(seed)}")

        parts.append(uuid4().hex[:8])
        return "-".join(parts)
    
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

        if model_key in {"panel_lstm", "panel_gru"}:
            return "panel"

        if model_key in {"arima", "random_forest", "rf"}:
            return "classical"

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
        self._write_json_payload(manifest_path, self._normalise(payload))

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

        self._write_json_payload(path, self._normalise(payload))

        return str(path)

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def append_run(self, record: RunRecord) -> None:
        payload = self._normalise(asdict(record))

        self._append_jsonl_payload(payload)

        self._append_csv_summary(payload)

        result_row = self._flatten_result_row(payload)
        self._append_flat_csv(self.run_results_csv_path, result_row)

        family = str(result_row["model_family"])
        family_csv_path = self.runs_dir / family / "run_results.csv"
        self._append_flat_csv(family_csv_path, result_row)

    def rebuild_run_result_csvs(self) -> Dict[str, Any]:
        """Rebuild analysis CSVs from the append-only JSONL experiment index."""

        if not self.jsonl_path.exists():
            return {"records": 0, "paths": {}}

        rows: list[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.jsonl_path} at line {line_number}"
                    ) from exc
                rows.append(self._flatten_result_row(payload))

        if not rows:
            return {"records": 0, "paths": {}}

        self._write_dict_rows_csv(self.run_results_csv_path, rows)
        paths: Dict[str, str] = {"all": str(self.run_results_csv_path)}

        grouped: Dict[str, list[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["model_family"]), []).append(row)

        for family, family_rows in grouped.items():
            path = self.runs_dir / family / "run_results.csv"
            self._write_dict_rows_csv(path, family_rows)
            paths[family] = str(path)

        return {"records": len(rows), "paths": paths}

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

        self._write_json_payload(path, payload)

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
            self._write_json_payload(
                group_summary_path,
                self._normalise(group_summary),
            )

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
            "k": row.get("k"),
            "graph_mode": row.get("graph_mode"),
            "graph_embed": row.get("graph_embed"),
            "graph_ablation": row.get("graph_ablation"),
            "ablate_feature": row.get("ablate_feature"),
            "seq_len": row.get("seq_len"),
            "batch_size": row.get("batch_size"),
            "lstm_epochs": row.get("lstm_epochs"),
            "stgnn_epochs": row.get("stgnn_epochs"),
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
                row.get("k"),
                row.get("graph_mode"),
                row.get("graph_embed"),
                row.get("graph_ablation"),
                row.get("ablate_feature"),
            )
            grouped.setdefault(key, []).append(row)

        out: list[Dict[str, Any]] = []

        for (
            queue_group,
            model,
            graph_model,
            prediction_window,
            k,
            graph_mode,
            graph_embed,
            graph_ablation,
            ablate_feature,
        ), group_rows in grouped.items():
            summary: Dict[str, Any] = {
                "queue_group": queue_group,
                "model": model,
                "graph_model": graph_model,
                "prediction_window": prediction_window,
                "k": k,
                "graph_mode": graph_mode,
                "graph_embed": graph_embed,
                "graph_ablation": graph_ablation,
                "ablate_feature": ablate_feature,
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
            "graph_ablation": payload.get("graph_ablation"),
            "ablate_feature": payload.get("ablate_feature"),
            "threshold_policy": payload.get("threshold_policy"),
            "deterministic": payload.get("deterministic"),
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

    def _flatten_result_row(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Stable, analysis-ready one-row representation of a completed run."""

        metrics = payload.get("metrics", {}) or {}
        extras = payload.get("extras", {}) or {}
        compute = extras.get("compute", {}) or {}
        metadata = extras.get("metadata", {}) or {}
        capacity = extras.get("capacity", {}) or metadata.get("capacity", {}) or {}
        graph_stats = extras.get("graph_stats", {}) or self._load_legacy_graph_stats(extras)
        random_forest = metadata.get("random_forest", {}) or {}
        arima_order = metadata.get("arima_order") or []
        model = str(payload.get("model") or "unknown").strip().lower()
        graph_model = str(payload.get("graph_model") or "").strip().lower()
        graph_backend = (
            extras.get("graph_backend")
            or metrics.get("graph_backend")
            or metadata.get("graph_backend")
        )
        family = self.model_group(model, graph_model)
        graph_stats_applicable = family in {"graph", "stgnn"}
        model_label = (
            f"stgnn+{graph_model}"
            if model == "stgnn" and graph_model
            else model
        )

        def metric(name: str, default=None):
            return self._normalise(metrics.get(name, default))

        def compute_value(name: str, metric_name: str | None = None):
            value = compute.get(name)
            if value is None:
                value = metadata.get(name)
            if value is None:
                value = metrics.get(metric_name or name)
            return self._normalise(value)

        def capacity_value(name: str, fallback=None):
            value = capacity.get(name)
            if value is None:
                value = fallback
            return self._normalise(value)

        def graph_value(name: str, *legacy_names):
            if not graph_stats_applicable:
                return None
            value = graph_stats.get(name)
            if value is None:
                for legacy_name in legacy_names:
                    value = graph_stats.get(legacy_name)
                    if value is not None:
                        break
            return self._normalise(value)

        def arima_order_value(index: int):
            try:
                return self._normalise(arima_order[index])
            except Exception:
                return None

        train_samples_value = compute_value("train_samples")
        lr_history = metadata.get("lr_history", []) or []
        inferred_epochs = len(lr_history) if isinstance(lr_history, list) else None
        is_arima = model == "arima"
        is_random_forest = model in {"random_forest", "rf"}
        is_neural = not is_arima and not is_random_forest

        training_sample_unit = compute_value("training_sample_unit")
        if training_sample_unit is None:
            training_sample_unit = (
                "raw_time_observation" if is_arima else "supervised_window"
            )

        train_examples_unique = compute_value("train_examples_unique")
        if train_examples_unique is None and train_samples_value is not None:
            if is_neural and inferred_epochs:
                train_examples_unique = int(
                    round(float(train_samples_value) / float(inferred_epochs))
                )
            else:
                train_examples_unique = train_samples_value

        sample_exposures = compute_value("sample_exposures")
        if sample_exposures is None and not is_arima:
            sample_exposures = train_samples_value

        epochs_completed = compute_value("epochs_completed")
        if epochs_completed is None and is_neural and inferred_epochs:
            epochs_completed = inferred_epochs

        total_parameters = compute_value("total_params")
        trainable_parameters = compute_value("trainable_params")
        inferred_capacity_family = (
            "arima" if is_arima else "random_forest" if is_random_forest else "neural"
        )
        inferred_primary_measure = None
        inferred_primary_value = None
        if is_neural:
            inferred_primary_measure = "trainable_parameters"
            inferred_primary_value = trainable_parameters
        elif is_random_forest and random_forest.get("n_estimators") is not None:
            inferred_primary_measure = "estimators"
            inferred_primary_value = random_forest.get("n_estimators")

        parameter_storage_bytes = capacity_value("parameter_storage_bytes")
        if parameter_storage_bytes is None and is_neural and total_parameters is not None:
            parameter_storage_bytes = int(total_parameters) * 4

        return {
            "run_id": payload.get("run_id"),
            "status": payload.get("status"),
            "timestamp_start": payload.get("timestamp_start"),
            "timestamp_end": payload.get("timestamp_end"),
            "duration_sec": payload.get("duration_sec"),
            "model_family": family,
            "model": model,
            "model_label": model_label,
            "graph_backend": graph_backend,
            "graph_model": graph_model or None,
            "ticker": payload.get("ticker"),
            "prediction_window": payload.get("prediction_window"),
            "horizon_bars": metric("horizon"),
            "seed": payload.get("seed"),
            "universe_id": payload.get("universe_id"),
            "interval": payload.get("interval"),
            "direction": payload.get("direction"),
            "confidence_pct": payload.get("confidence"),
            "threshold_policy": payload.get("threshold_policy"),
            "deterministic": payload.get("deterministic"),
            "threshold_operational": metric(
                "threshold_operational",
                metric("threshold_fixed"),
            ),
            "threshold_validation_macro_f1": metric("threshold_macro_f1_dense"),
            "accuracy_dense": metric("accuracy_dense"),
            "f1_positive_dense": metric("f1_dense"),
            "macro_f1_dense": metric("macro_f1_dense"),
            "roc_auc_dense": metric("roc_auc_dense"),
            "average_precision_dense": metric("ap_dense"),
            "validation_loss": metric("val_loss_dense"),
            "test_loss": metric("test_loss_dense"),
            "n_predictions_dense": metric("n_predictions_dense"),
            "accuracy_fixed_05": metric("accuracy_dense_fixed_05"),
            "f1_positive_fixed_05": metric("f1_dense_fixed_05"),
            "macro_f1_fixed_05": metric("macro_f1_dense_fixed_05"),
            "accuracy_trade_aligned": metric("accuracy_trade_aligned"),
            "f1_positive_trade_aligned": metric("f1_trade_aligned"),
            "macro_f1_trade_aligned": metric("macro_f1_trade_aligned"),
            "roc_auc_trade_aligned": metric("roc_auc_trade_aligned"),
            "average_precision_trade_aligned": metric("ap_trade_aligned"),
            "n_trades": metric("n_trades"),
            "hit_rate": metric("hit_rate"),
            "mean_trade_return": metric("mean_trade_return"),
            "sharpe": metric("sharpe"),
            "final_equity": metric("final_equity"),
            "max_drawdown": metric("max_drawdown"),
            "train_seconds": compute_value("train_seconds"),
            "energy_wh": compute_value("energy_wh"),
            "average_power_w": compute_value("avg_power_w"),
            "peak_gpu_memory_mb": compute_value("gpu_peak_memory_mb"),
            "train_samples": train_samples_value,
            "training_sample_unit": training_sample_unit,
            "train_examples_unique": train_examples_unique,
            "sample_exposures": sample_exposures,
            "epochs_completed": epochs_completed,
            "energy_per_sample_wh": compute_value("energy_per_sample_wh"),
            "energy_measurement_method": compute_value("energy_measurement_method"),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "capacity_family": capacity_value("family", inferred_capacity_family),
            "capacity_primary_measure": capacity_value(
                "primary_measure",
                inferred_primary_measure,
            ),
            "capacity_primary_value": capacity_value(
                "primary_value",
                inferred_primary_value,
            ),
            "parameter_storage_bytes": parameter_storage_bytes,
            "neural_total_parameters": capacity_value(
                "neural_total_parameters",
                total_parameters if is_neural else None,
            ),
            "neural_trainable_parameters": capacity_value(
                "neural_trainable_parameters",
                trainable_parameters if is_neural else None,
            ),
            "rf_estimators": capacity_value(
                "rf_estimators",
                random_forest.get("n_estimators"),
            ),
            "rf_total_nodes": capacity_value(
                "rf_total_nodes",
                random_forest.get("total_nodes"),
            ),
            "rf_total_leaves": capacity_value(
                "rf_total_leaves",
                random_forest.get("total_leaves"),
            ),
            "rf_max_depth_observed": capacity_value(
                "rf_max_depth_observed",
                random_forest.get("max_depth_observed"),
            ),
            "rf_mean_depth": capacity_value(
                "rf_mean_depth",
                random_forest.get("mean_depth"),
            ),
            "rf_n_features_in": capacity_value(
                "rf_n_features_in",
                random_forest.get("n_features_in"),
            ),
            "arima_p": capacity_value("arima_p", arima_order_value(0)),
            "arima_d": capacity_value("arima_d", arima_order_value(1)),
            "arima_q": capacity_value("arima_q", arima_order_value(2)),
            "arima_parameter_count": capacity_value("arima_parameter_count"),
            "arima_state_dimension": capacity_value("arima_state_dimension"),
            "arima_aic": capacity_value("arima_aic"),
            "arima_bic": capacity_value("arima_bic"),
            "graph_stats_applicable": graph_stats_applicable,
            "graph_requested_k": graph_value("requested_k"),
            "graph_effective_k": graph_value("effective_k"),
            "graph_num_nodes": graph_value("num_nodes"),
            "graph_num_edges": graph_value("num_edges"),
            "graph_num_mst_edges": graph_value("num_mst_edges"),
            "graph_density": graph_value("density"),
            "graph_mean_degree": graph_value("mean_degree"),
            "graph_connected_components": graph_value("connected_components"),
            "graph_isolated_nodes": graph_value("isolated_nodes"),
            "sector_edge_homophily": graph_value(
                "sector_edge_homophily",
                "homophily",
            ),
            "sector_homophilous_edges": graph_value("sector_homophilous_edges"),
            "sector_homophily_eligible_edges": graph_value(
                "sector_homophily_eligible_edges"
            ),
            "sector_known_nodes": graph_value("sector_known_nodes"),
            "sector_unknown_nodes": graph_value("sector_unknown_nodes"),
            "k": payload.get("k"),
            "graph_mode": payload.get("graph_mode"),
            "graph_embed": payload.get("graph_embed"),
            "graph_ablation": payload.get("graph_ablation"),
            "ablate_feature": payload.get("ablate_feature"),
            "job_id": payload.get("job_id"),
            "queue_run_id": payload.get("queue_run_id"),
            "error_message": extras.get("error_message"),
        }

    def _load_legacy_graph_stats(self, extras: Dict[str, Any]) -> Dict[str, Any]:
        """Recover graph metadata from per-run JSON written before CSV support."""

        for key, nested_key in (
            ("graph_stats_path", None),
            ("canonical_result_path", "graph_stats"),
        ):
            raw_path = extras.get(key)
            if not raw_path:
                continue
            path = Path(str(raw_path))
            candidates = [path]
            if not path.is_absolute():
                candidates.append(Path.cwd() / path)
                root_dir = getattr(self, "root_dir", None)
                if root_dir is not None:
                    candidates.extend(
                        [Path(root_dir) / path, Path(root_dir).parent / path]
                    )
            for candidate in candidates:
                if not candidate.exists():
                    continue
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if nested_key is not None:
                    payload = payload.get(nested_key, {}) if isinstance(payload, dict) else {}
                if isinstance(payload, dict):
                    return payload
        return {}

    def _append_flat_csv(self, path: Path, row: Dict[str, Any]) -> None:
        """Append a row, expanding an older header without dropping data."""

        path.parent.mkdir(parents=True, exist_ok=True)
        normalised = {key: self._normalise(value) for key, value in row.items()}
        desired_header = list(normalised.keys())

        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=desired_header)
                writer.writeheader()
                writer.writerow(normalised)
            return

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, [])

        if existing_header == desired_header:
            with path.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=desired_header).writerow(normalised)
            return

        with path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))

        fieldnames = list(dict.fromkeys(existing_header + desired_header))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow({key: existing.get(key, "") for key in fieldnames})
            writer.writerow({key: normalised.get(key, "") for key in fieldnames})

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

        if isinstance(value, float) and not math.isfinite(value):
            return None

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)
    
    def run_dir(
        self,
        run_id: str,
        *,
        model: str = "",
        graph_backend: str | None = None,
    ) -> Path:
        model_key = str(model or "unknown").strip().lower() or "unknown"
        backend_key = str(graph_backend or "").strip().lower()

        group = self.model_group(model_key, backend_key)

        if model_key == "stgnn" and backend_key:
            path = self.runs_dir / group / model_key / backend_key / str(run_id)
        else:
            path = self.runs_dir / group / model_key / str(run_id)

        path.mkdir(parents=True, exist_ok=True)
        return path


    def save_run_payload(
        self,
        *,
        run_id: str,
        filename: str,
        model: str = "",
        graph_backend: str | None = None,
        payload: Any,
    ) -> str:
        path = self.run_dir(
            run_id,
            model=model,
            graph_backend=graph_backend,
        ) / filename

        self._write_json_payload(path, self._normalise(payload))

        return str(path)
