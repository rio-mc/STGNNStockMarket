from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from evaluation.evaluation_types import EvaluationResult

from ..base_runner import BaseModelRunner, ModelRunResult


class RandomForestRunner(BaseModelRunner):
    model_name = "RANDOM_FOREST"

    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        app.frontendApp.set_status("Training Random Forest baseline...")

        train_ds, val_ds, test_ds, _train_df_stock, _val_df_stock, _test_df_stock = self._build_recurrent_datasets(app, stock)
        self._log_dataset_summary(app, train_ds, val_ds, test_ds, getattr(train_ds, "aligned_tickers", [stock]))

        model = self._make_model(app)
        x_train, y_train = self._dataset_to_numpy(train_ds)

        start = time.time()
        model.fit(x_train, y_train)
        train_seconds = time.time() - start

        validation_result = self._evaluate_dataset(
            app=app,
            model=model,
            dataset=val_ds,
            split_name="validation",
            decision_threshold=0.5,
            stop_event=stop_event,
        )
        threshold_selection = self._calibrate_threshold(app, validation_result)
        eval_result = self._evaluate_dataset(
            app=app,
            model=model,
            dataset=test_ds,
            split_name="test",
            decision_threshold=threshold_selection["selected_threshold"],
            stop_event=stop_event,
        )
        self._attach_metadata(app, eval_result, stock=stock)
        energy_metadata = self._cpu_energy_metadata(
            app,
            train_seconds=train_seconds,
            train_samples=len(train_ds),
        )
        capacity = self._random_forest_capacity_metadata(model)
        eval_result.metadata.update({
            "threshold_source": "validation",
            "validation_selection": threshold_selection,
            "validation_loss_dense": threshold_selection["dense_loss"],
            "test_loss_dense": getattr(eval_result, "dense_val_loss", None),
            "threshold_macro_f1_validation": threshold_selection["macro_f1_threshold"],
            **energy_metadata,
            "train_seconds": float(train_seconds),
            "train_samples": int(len(train_ds)),
            "training_sample_unit": "supervised_window",
            "train_examples_unique": int(len(train_ds)),
            "sample_exposures": int(len(train_ds)),
            "epochs_completed": None,
            "gpu_peak_memory_mb": None,
            "total_params": None,
            "trainable_params": None,
            "capacity": capacity,
            "random_forest": {
                "n_estimators": int(getattr(model, "n_estimators", 0)),
                "max_depth": getattr(model, "max_depth", None),
                "min_samples_leaf": getattr(model, "min_samples_leaf", None),
                "class_weight": getattr(model, "class_weight", None),
                "n_features_in": int(getattr(model, "n_features_in_", 0)),
                "total_nodes": capacity["rf_total_nodes"],
                "total_leaves": capacity["rf_total_leaves"],
                "max_depth_observed": capacity["rf_max_depth_observed"],
                "mean_depth": capacity["rf_mean_depth"],
            },
        })

        metrics = evaluator.evaluate(
            model_name=self.model_name,
            result=eval_result,
            price_df=price_df,
        )

        app._check_stop(stop_event)
        app.frontendApp.set_status("Predicting with Random Forest baseline...")

        live_prob = self._live_probability(model, test_ds)
        threshold = self._resolve_threshold(
            metrics,
            policy=getattr(app.args, "decision_threshold_policy", "macro_f1_dense"),
        )
        direction, confidence = self._direction_from_probability(live_prob, threshold)

        app.logger.info("[RandomForestRunner] %s (%.1f%%)", direction, confidence)

        self._cleanup(train_ds, val_ds, test_ds)
        return ModelRunResult(
            model_name=self.model_name,
            direction=direction,
            confidence=confidence,
            metrics=metrics,
            eval_result=eval_result,
            trainer=None,
            model=model,
            extras={"random_forest": eval_result.metadata["random_forest"]},
        )

    def _make_model(self, app):
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:
            raise RuntimeError(
                "Random Forest baseline requires scikit-learn. Install project dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from exc

        max_depth = getattr(app.args, "rf_max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)
            if max_depth <= 0:
                max_depth = None

        class_weight = None
        if str(getattr(app.args, "class_balance", "auto")).strip().lower() == "auto":
            class_weight = "balanced_subsample"

        return RandomForestClassifier(
            n_estimators=int(getattr(app.args, "rf_estimators", 300)),
            max_depth=max_depth,
            min_samples_leaf=int(getattr(app.args, "rf_min_samples_leaf", 5)),
            class_weight=class_weight,
            random_state=int(getattr(app.args, "seed", 42)),
            n_jobs=int(getattr(app.args, "rf_n_jobs", -1)),
        )

    @staticmethod
    def _random_forest_capacity_metadata(model) -> dict:
        estimators = list(getattr(model, "estimators_", []) or [])
        total_nodes = 0
        total_leaves = 0
        depths = []
        storage_bytes = 0

        for estimator in estimators:
            tree = getattr(estimator, "tree_", None)
            if tree is None:
                continue

            total_nodes += int(getattr(tree, "node_count", 0) or 0)
            total_leaves += int(getattr(tree, "n_leaves", 0) or 0)
            depths.append(int(getattr(tree, "max_depth", 0) or 0))

            try:
                state = tree.__getstate__()
                storage_bytes += int(
                    sum(
                        int(getattr(value, "nbytes", 0) or 0)
                        for value in state.values()
                    )
                )
            except Exception:
                pass

        mean_depth = float(np.mean(depths)) if depths else None
        max_depth = max(depths) if depths else None
        n_estimators = int(len(estimators) or getattr(model, "n_estimators", 0) or 0)

        return {
            "family": "random_forest",
            "primary_measure": "tree_nodes",
            "primary_value": int(total_nodes),
            "parameter_storage_bytes": int(storage_bytes),
            "rf_estimators": n_estimators,
            "rf_total_nodes": int(total_nodes),
            "rf_total_leaves": int(total_leaves),
            "rf_max_depth_observed": max_depth,
            "rf_mean_depth": mean_depth,
            "rf_n_features_in": int(getattr(model, "n_features_in_", 0) or 0),
        }

    @staticmethod
    def _dataset_to_numpy(dataset):
        x = dataset.x.detach().cpu().numpy().astype(np.float32)
        y = dataset.y.detach().cpu().numpy().astype(int)
        return x.reshape(x.shape[0], -1), y

    def _evaluate_dataset(
        self,
        *,
        app,
        model,
        dataset,
        split_name: str,
        decision_threshold: float,
        stop_event,
    ) -> EvaluationResult:
        x_val, y_val = self._dataset_to_numpy(dataset)
        probs = self._predict_proba(model, x_val)
        preds = (probs >= float(decision_threshold)).astype(int)

        y_true_all = []
        y_pred_all = []
        probs_all = []
        prediction_dates = []
        hist_val = []

        for i, (truth, pred, prob) in enumerate(zip(y_val, preds, probs)):
            app._check_stop(stop_event)
            timestamp = pd.Timestamp(dataset.get_timestamp(i)).tz_localize(None)
            loss = self._binary_cross_entropy(float(prob), int(truth))

            y_true_all.append(int(truth))
            y_pred_all.append(int(pred))
            probs_all.append(float(prob))
            prediction_dates.append(timestamp)
            hist_val.append({"date": timestamp, "loss": loss})

            if hasattr(app.frontendApp, "updateProgress"):
                app.frontendApp.updateProgress((i + 1) / max(len(dataset), 1))

        mean_val_loss = float(np.mean([row["loss"] for row in hist_val])) if hist_val else float("nan")

        return EvaluationResult(
            y_true=y_true_all,
            y_pred=y_pred_all,
            probs=probs_all,
            prediction_dates=prediction_dates,
            decision_threshold=float(decision_threshold),
            dense_val_loss=mean_val_loss,
            hist_train=[],
            hist_val=hist_val,
            horizon=int(app.horizon),
            model_name=self.model_name,
            metadata={
                "evaluation_mode": "dense_rolling",
                "evaluation_split": str(split_name),
                "decision_threshold_policy": "fixed",
                "ticker": getattr(app.args, "target_stock", None),
            },
        )

    @staticmethod
    def _predict_proba(model, x: np.ndarray) -> np.ndarray:
        classes = list(getattr(model, "classes_", []))
        raw = model.predict_proba(x)
        if 1 in classes:
            return raw[:, classes.index(1)].astype(float)
        if raw.shape[1] == 1:
            only_class = int(classes[0]) if classes else 0
            return np.ones(x.shape[0], dtype=float) if only_class == 1 else np.zeros(x.shape[0], dtype=float)
        return raw[:, -1].astype(float)

    def _live_probability(self, model, val_ds) -> float:
        x = val_ds.x[-1:].detach().cpu().numpy().astype(np.float32)
        x = x.reshape(x.shape[0], -1)
        return float(self._predict_proba(model, x)[0])

    @staticmethod
    def _binary_cross_entropy(prob: float, truth: int) -> float:
        p = min(max(float(prob), 1e-7), 1.0 - 1e-7)
        y = int(truth)
        return float(-(y * math.log(p) + (1 - y) * math.log(1.0 - p)))
