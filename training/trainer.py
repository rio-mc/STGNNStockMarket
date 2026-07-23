import time
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
try:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlInit,
        nvmlShutdown,
    )
except ImportError:
    nvmlDeviceGetHandleByIndex = None
    nvmlDeviceGetPowerUsage = None
    nvmlInit = None
    nvmlShutdown = None
try:
    from torch_geometric.data import Batch as GeoBatch, Data
except ImportError:
    GeoBatch = None
    Data = None

from evaluation.evaluation_types import EvaluationResult


class Trainer:
    """
    Handles training and rolling evaluation for a single active model run.
    """

    def __init__(
        self,
        model,
        optimiser,
        criterion,
        device,
        graphBuilder=None,
        features=None,
        tickers=None,
        targetTicker=None,
        frontend=None,
        evaluator=None,
        prediction_horizon=None,
        seq_len=None,
        model_name=None,
        weight_decay=None,
        lr_scheduler="reduce_on_plateau",
        lr_plateau_factor=0.5,
        lr_plateau_patience=5,
        lr_plateau_min_lr=0.0,
        cpu_power_watts=None,
        training_log="summary",
    ):
        self.frontend = frontend
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.device = device
        self.evaluator = evaluator

        self.features = features or {}
        self.tickers = tickers or []
        self.prediction_horizon = prediction_horizon
        self.seq_len = seq_len

        self.graphBuilder = graphBuilder
        self.targetTicker = targetTicker
        if self.graphBuilder is not None and hasattr(self.graphBuilder, "edge_index"):
            self.edge_index = self.graphBuilder.edge_index.to(self.device)
        else:
            self.edge_index = None

        self.targetIdx = self._resolve_target_index()
        self.model.edge_index = self.edge_index
        self.model.target_node_index = self.targetIdx
        self.model_name = model_name

        self.weight_decay = float(weight_decay) if weight_decay is not None else 0.0
        self.lr_scheduler_name = str(lr_scheduler or "none").strip().lower()
        self.lr_plateau_factor = float(lr_plateau_factor)
        self.lr_plateau_patience = int(lr_plateau_patience)
        self.lr_plateau_min_lr = float(lr_plateau_min_lr)
        self.cpu_power_watts = float(cpu_power_watts) if cpu_power_watts is not None else None
        self.training_log = str(training_log or "summary").strip().lower()
        if self.training_log not in {"quiet", "summary", "epochs"}:
            raise ValueError("training_log must be one of: quiet, summary, epochs")
        self.energy_measurement_method = "unavailable"
        self.clip_max_norm = 1.0
        self._clip_activations = 0
        self.decision_threshold = 0.5
        self.decision_threshold_policy = "fixed"

        # Single-model history buffers
        self.train_loss_history = []
        self.val_loss_history = []
        self.epoch_val_loss_history = []
        self.rolling_loss_history = {}
        self.lr_history = []

    def train(
        self,
        dataloader,
        num_epochs,
        validation_dataloader=None,
        stop_event=None,
        patience=15,
    ):
        self._init_energy_monitoring()

        total_energy_Wh = 0.0
        total_train_seconds = 0.0
        self.energy_epochs_Wh = []
        self.energy_per_sample_epochs_Wh = []
        self.samples_per_epoch = []
        total_samples = 0
        self.total_energy_Wh = 0.0
        self.total_train_seconds = 0.0
        self.avg_power_W = 0.0

        scheduler = self._make_scheduler()

        epoch_bar = tqdm(
            range(num_epochs),
            desc="Training",
            position=0,
            disable=self.training_log != "epochs",
        )

        best_monitor_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None
        min_delta = 1e-6
        monitor_name = "validation" if validation_dataloader is not None else "training"

        for epoch in epoch_bar:
            epoch_samples = 0
            self.model.train()
            epoch_losses = []
            epoch_start_time = time.time()
            power_samples = []

            for batch_idx, batch in enumerate(dataloader):
                try:

                    power_samples.append(self._get_power_usage())

                    x, y, edge_index, edge_attr = self._unpack_batch(batch)

                    batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1
                    epoch_samples += int(batch_size)

                    self.optimiser.zero_grad(set_to_none=True)
                    loss = self._forward_and_loss(x, y, edge_index, edge_attr)
                    loss.backward()
                    self._clip_grads()
                    self.optimiser.step()

                    epoch_losses.append(loss.item())

                    progress = ((epoch * len(dataloader)) + batch_idx + 1) / (num_epochs * len(dataloader))
                    if self.frontend is not None:
                        self.frontend.updateProgress(progress)

                    if stop_event and stop_event.is_set():
                        print("[Trainer] Early stopping triggered.")
                        time.sleep(0.01)
                        if getattr(self.device, "type", None) == "cuda":
                            torch.cuda.synchronize()
                        self._finalise_energy_monitoring()
                        return

                except Exception as e:
                    print(f"[Trainer][FATAL] Batch failed at epoch={epoch} batch={batch_idx}: {e}")
                    raise

            epoch_time = time.time() - epoch_start_time
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            total_train_seconds += epoch_time

            avg_power = sum(power_samples) / max(len(power_samples), 1)
            energy_Wh = avg_power * (epoch_time / 3600.0)
            total_energy_Wh += energy_Wh

            self.energy_epochs_Wh.append(energy_Wh)
            total_samples += epoch_samples
            self.samples_per_epoch.append(epoch_samples)

            energy_per_sample_Wh = (energy_Wh / epoch_samples) if epoch_samples > 0 else 0.0
            self.energy_per_sample_epochs_Wh.append(energy_per_sample_Wh)

            val_loss = self._mean_loss(validation_dataloader)
            monitor_loss = val_loss if val_loss is not None else avg_loss
            monitor_name = "validation" if val_loss is not None else "training"

            if val_loss is not None:
                self.epoch_val_loss_history.append(float(val_loss))

            if scheduler is not None:
                scheduler.step(monitor_loss)

            current_lr = self._current_lr()
            self.lr_history.append({
                "epoch": int(epoch + 1),
                "lr": float(current_lr),
                "train_loss": float(avg_loss),
                "val_loss": float(val_loss) if val_loss is not None else None,
            })

            # Store training loss locally for this single active model run
            self.train_loss_history.append(float(avg_loss))

            if monitor_loss < best_monitor_loss - min_delta:
                best_monitor_loss = float(monitor_loss)
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if self.training_log != "quiet":
                    tqdm.write(
                        f"[Training] early_stop epoch={epoch + 1} "
                        f"best_{monitor_name}_loss={best_monitor_loss:.6f} "
                        f"patience={patience}"
                    )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if self.energy_measurement_method == "unavailable":
            self.total_energy_Wh = None
            self.avg_power_W = None
            self.energy_per_sample_Wh = None
        else:
            self.total_energy_Wh = total_energy_Wh
            self.avg_power_W = (total_energy_Wh * 3600.0 / total_train_seconds) if total_train_seconds > 0 else 0.0
            self.energy_per_sample_Wh = (total_energy_Wh / total_samples) if total_samples > 0 else 0.0
        self.total_train_seconds = total_train_seconds
        self.total_samples = total_samples

        if self.training_log != "quiet":
            energy = (
                "unavailable"
                if self.total_energy_Wh is None
                else f"{self.total_energy_Wh:.4f} Wh"
            )
            tqdm.write(
                f"[Training] epochs={len(self.train_loss_history)} "
                f"best_{monitor_name}_loss={best_monitor_loss:.6f} "
                f"energy={energy} clipped_batches={self._clip_activations}"
            )

        self._finalise_energy_monitoring()
        return self.total_energy_Wh

    def evaluate_rolling(self, val_dataset, *, split_name="validation"):
        self.model.eval()

        total_samples = len(val_dataset)
        y_true_all, y_pred_all, probs_all, prediction_dates = [], [], [], []
        losses = []
        split_loss_history = []

        with torch.no_grad():
            for i in range(0, total_samples, 1):
                try:
                    sample = val_dataset[i]
                    timestamp = None
                    edge_index = None
                    edge_attr = None

                    if Data is not None and isinstance(sample, Data):
                        x = sample.x.unsqueeze(0).to(self.device)
                        y = sample.y.unsqueeze(0).to(self.device)
                        edge_index = sample.edge_index.to(self.device)
                        if hasattr(sample, "edge_attr") and sample.edge_attr is not None:
                            edge_attr = sample.edge_attr.to(self.device)

                    elif isinstance(sample, tuple):
                        x, y = sample
                        x = x.unsqueeze(0).to(self.device)
                        y = y.unsqueeze(0).to(self.device)

                    else:
                        print(f"[WARNING] Unrecognised sample format at index={i}, skipping.")
                        continue

                    timestamp = val_dataset.get_timestamp(i)
                    if timestamp:
                        timestamp = pd.Timestamp(timestamp).tz_localize(None)
                        prediction_dates.append(timestamp)

                    if self.graphBuilder:
                        logits = self.model(
                            x,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            target_node_index=self.targetIdx,
                        )
                    else:
                        if self.targetIdx is not None:
                            logits = self.model(x, target_node_index=self.targetIdx)
                        else:
                            logits = self.model(x)

                    prob = torch.sigmoid(logits).item()
                    pred = int(prob >= self.decision_threshold)
                    truth = int(y.item())

                    y_pred_all.append(pred)
                    y_true_all.append(truth)
                    probs_all.append(prob)

                    if y.shape != logits.shape:
                        y = y.view_as(logits)
                    loss = self.criterion(logits, y)
                    losses.append(loss.item())

                    split_loss_history.append({
                        "date": timestamp,
                        "loss": float(loss.item()),
                    })

                except Exception as e:
                    print(f"[ERROR] Skipping sample index={i} due to: {e}")
                    continue

        mean_val_loss = float(np.mean(losses)) if losses else float("nan")
        split_label = str(split_name or "evaluation").strip().upper()
        self.rolling_loss_history[str(split_name)] = split_loss_history
        if str(split_name).strip().lower() == "validation":
            self.val_loss_history = split_loss_history

        print(f"[{split_label}][SUMMARY] total_samples={total_samples}, predictions={len(prediction_dates)}")
        print(f"[{split_label}] Predicted 1s: {sum(y_pred_all)}, 0s: {len(y_pred_all) - sum(y_pred_all)}")
        print(f"[{split_label}] True 1s: {sum(y_true_all)}, 0s: {len(y_true_all) - sum(y_true_all)}")
        if probs_all:
            probs_np = np.asarray(probs_all, dtype=float)
            print(
                f"[{split_label}] Probability summary: "
                f"min={probs_np.min():.3f}, mean={probs_np.mean():.3f}, "
                f"max={probs_np.max():.3f}, std={probs_np.std():.3f}"
            )
        print(f"[{split_label}] Mean loss (dense): {mean_val_loss:.6f}")

        return EvaluationResult(
            y_true=y_true_all,
            y_pred=y_pred_all,
            probs=probs_all,
            prediction_dates=prediction_dates,
            decision_threshold=float(self.decision_threshold),
            dense_val_loss=mean_val_loss,
            hist_train=self.train_loss_history,
            hist_val=split_loss_history,
            horizon=self.prediction_horizon,
            model_name=self.model_name,
            metadata={
                "evaluation_mode": "dense_rolling",
                "evaluation_split": str(split_name),
                "decision_threshold_policy": self.decision_threshold_policy,
                "ticker": self.targetTicker,
                "lr_scheduler": self.lr_scheduler_name,
                "lr_history": self.lr_history,
                "epoch_val_loss_history": self.epoch_val_loss_history,
            },
        )

    def _mean_loss(self, dataloader):
        """Return sample-weighted mean loss without changing model parameters."""
        if dataloader is None:
            return None

        self.model.eval()
        weighted_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                x, y, edge_index, edge_attr = self._unpack_batch(batch)
                loss = self._forward_and_loss(x, y, edge_index, edge_attr)
                batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1
                weighted_loss += float(loss.item()) * int(batch_size)
                total_samples += int(batch_size)

        if total_samples == 0:
            return None
        return weighted_loss / total_samples

    def _unpack_batch(self, batch):
        if GeoBatch is not None and isinstance(batch, GeoBatch):
            if batch.x.dim() != 3:
                raise ValueError(
                    f"[Trainer:_unpack_batch] Expected batch.x dim=3, got shape={tuple(batch.x.shape)}"
                )

            num_graphs = int(batch.num_graphs)
            num_nodes = len(self.tickers)
            seq_len = batch.x.size(1)
            feat_dim = batch.x.size(2)

            expected_rows = num_graphs * num_nodes
            if batch.x.size(0) != expected_rows:
                raise ValueError(
                    f"[Trainer:_unpack_batch] batch.x first dim mismatch: "
                    f"got {batch.x.size(0)}, expected {expected_rows} "
                    f"(num_graphs={num_graphs}, num_nodes={num_nodes})"
                )

            x = batch.x.contiguous().view(num_graphs, num_nodes, seq_len, feat_dim)
            y = batch.y.view(num_graphs, -1)

            edge_index = batch.edge_index
            edge_attr = getattr(batch, "edge_attr", None)

            edge_index = edge_index.to(self.device)
            if edge_index.dtype != torch.long:
                edge_index = edge_index.to(torch.long)
            if edge_index.dim() == 1:
                if edge_index.numel() % 2 != 0:
                    raise ValueError("edge_index 1-D length must be even")
                edge_index = edge_index.view(2, -1).contiguous()
            elif edge_index.dim() == 2 and edge_index.size(0) != 2:
                edge_index = edge_index.t().contiguous()

            edge_attr = edge_attr.to(self.device) if edge_attr is not None else None
            return x.to(self.device), y.to(self.device), edge_index, edge_attr

        if isinstance(batch, (tuple, list)):
            x, y = batch
            return x.to(self.device), y.to(self.device), None, None

        raise ValueError(f"[Trainer:_unpack_batch] Unexpected batch type: {type(batch)}")

    def _forward_and_loss(self, x, y, edge_index, edge_attr=None):
        if self.graphBuilder:
            assert self.targetIdx is not None, "STGNN requires a valid target index"
            pred = self.model(
                x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                target_node_index=self.targetIdx,
            )
        else:
            if self.targetIdx is not None:
                pred = self.model(x, target_node_index=self.targetIdx)
            else:
                pred = self.model(x)

        y = y.view(-1, 1)
        loss = self.criterion(pred, y)
        return loss

    def _resolve_target_index(self):
        if self.graphBuilder and self.targetTicker in self.tickers:
            return self.tickers.index(self.targetTicker)
        return None

    def _init_energy_monitoring(self):
        if getattr(self.device, "type", None) != "cuda":
            self.handle = None
            if self.cpu_power_watts is not None:
                self.energy_measurement_method = "cpu_power_estimate"
            else:
                self.energy_measurement_method = "unavailable_cpu_power_not_configured"
            return
        if nvmlInit is None:
            print("[EnergyMonitor] pynvml not installed; GPU power tracking disabled.")
            self.handle = None
            self.energy_measurement_method = "unavailable_nvml_not_installed"
            return

        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
            self.energy_measurement_method = "nvml_gpu_power"
        except Exception as e:
            print(f"[EnergyMonitor] Failed to init NVML: {e}")
            self.handle = None
            self.energy_measurement_method = "unavailable_nvml_init_failed"

    def _get_power_usage(self):
        if self.handle:
            try:
                return nvmlDeviceGetPowerUsage(self.handle) / 1000
            except Exception:
                return 0.0
        if self.energy_measurement_method == "cpu_power_estimate" and self.cpu_power_watts is not None:
            return float(self.cpu_power_watts)
        return 0.0

    def _finalise_energy_monitoring(self):
        try:
            if getattr(self, "handle", None) is not None and nvmlShutdown is not None:
                nvmlShutdown()
        except Exception:
            pass

    def _clip_grads(self):
        if not self.clip_max_norm or self.clip_max_norm <= 0:
            return 0.0
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
        try:
            engaged = float(total_norm) > float(self.clip_max_norm)
        except Exception:
            engaged = False
        if engaged:
            self._clip_activations += 1
        return float(total_norm)

    def _make_scheduler(self):
        if self.lr_scheduler_name in {"none", "off", "false"}:
            return None
        if self.lr_scheduler_name not in {"reduce_on_plateau", "reducelronplateau"}:
            raise ValueError(
                "lr_scheduler must be one of: reduce_on_plateau, none"
            )
        if not 0.0 < self.lr_plateau_factor < 1.0:
            raise ValueError("lr_plateau_factor must be > 0 and < 1.")
        if self.lr_plateau_patience < 0:
            raise ValueError("lr_plateau_patience must be >= 0.")
        if self.lr_plateau_min_lr < 0.0:
            raise ValueError("lr_plateau_min_lr must be >= 0.")

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            factor=self.lr_plateau_factor,
            patience=self.lr_plateau_patience,
            min_lr=self.lr_plateau_min_lr,
        )

    def _current_lr(self) -> float:
        if not self.optimiser.param_groups:
            return 0.0
        return float(self.optimiser.param_groups[0].get("lr", 0.0))
