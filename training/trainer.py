import time
import torch
import numpy as np
from torch_geometric.data import Batch as GeoBatch, Data
import pandas as pd
from evaluation.evaluation_types import EvaluationResult
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown


class Trainer:
    """
    Handles training and evaluation for both LSTM and STGNN models.
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
        self.clip_max_norm = 1.0
        self._clip_activations = 0
        self.decision_threshold = 0.5

    def train(self, dataloader, num_epochs, stop_event=None):
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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            factor=0.5,
            patience=5
        )

        for epoch in range(num_epochs):
            epoch_samples = 0
            self.model.train()
            epoch_losses = []
            epoch_start_time = time.time()
            power_samples = []

            print(f"[Trainer] Starting epoch {epoch + 1}/{num_epochs}")

            for batch_idx, batch in enumerate(dataloader):
                try:
                    if batch_idx == 0:
                        print(f"[Trainer] First batch type: {type(batch)}")

                    power_samples.append(self._get_gpu_power_usage())

                    x, y, edge_index, edge_attr = self._unpack_batch(batch)

                    if batch_idx == 0:
                        print(f"[Trainer] x shape: {tuple(x.shape)}")
                        print(f"[Trainer] y shape: {tuple(y.shape)}")
                        if edge_index is not None:
                            print(f"[Trainer] edge_index shape: {tuple(edge_index.shape)}")
                        if edge_attr is not None:
                            print(f"[Trainer] edge_attr shape: {tuple(edge_attr.shape)}")

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
                        if torch.cuda.is_available():
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

            print(f"[Epoch {epoch}] Avg Loss = {avg_loss:.4f}")
            print(f"[Epoch {epoch}] Duration = {epoch_time:.2f}s | Avg Power = {avg_power:.2f} W | Energy = {energy_Wh:.4f} Wh")

            scheduler.step(avg_loss)

            model_key = self.model_name
            self.evaluator.record_training_loss(model_key, avg_loss)

            self.evaluator.plot_loss(
                hist_l=self.evaluator.get_training_loss("LSTM"),
                hist_s=self.evaluator.get_training_loss("STGNN"),
                hist_g=self.evaluator.get_training_loss("GRU"),
                val_l=[v["loss"] for v in self.evaluator.get_validation_loss("LSTM")],
                val_s=[v["loss"] for v in self.evaluator.get_validation_loss("STGNN")],
                val_g=[v["loss"] for v in self.evaluator.get_validation_loss("GRU")],
            )

        self.total_energy_Wh = total_energy_Wh
        self.total_train_seconds = total_train_seconds
        self.avg_power_W = (total_energy_Wh * 3600.0 / total_train_seconds) if total_train_seconds > 0 else 0.0
        self.total_samples = total_samples
        self.energy_per_sample_Wh = (total_energy_Wh / total_samples) if total_samples > 0 else 0.0

        print(f"[Training Summary] Total energy used: {self.total_energy_Wh:.4f} Wh")
        print(f"[Training Summary] Batches with gradient clipping: {self._clip_activations}")

        self._finalise_energy_monitoring()
        return self.total_energy_Wh

    def evaluate_rolling(self, val_dataset):
        self.model.eval()

        total_samples = len(val_dataset)
        y_true_all, y_pred_all, probs_all, prediction_dates = [], [], [], []
        losses = []

        with torch.no_grad():
            for i in range(0, total_samples, 1):
                try:
                    sample = val_dataset[i]
                    timestamp = None
                    edge_index = None
                    edge_attr = None

                    if isinstance(sample, Data):
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
                            target_node_index=self.targetIdx
                        )
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

                    self.evaluator.record_validation_loss(self.model_name, loss.item(), timestamp)

                except Exception as e:
                    print(f"[ERROR] Skipping sample index={i} due to: {e}")
                    continue

        mean_val_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[EVAL][SUMMARY] total_samples={total_samples}, predictions={len(prediction_dates)}")
        print(f"[EVAL] Predicted 1s: {sum(y_pred_all)}, 0s: {len(y_pred_all) - sum(y_pred_all)}")
        print(f"[EVAL] True 1s: {sum(y_true_all)}, 0s: {len(y_true_all) - sum(y_true_all)}")
        print(f"[EVAL] Mean validation loss (dense): {mean_val_loss:.6f}")

        val_l = [v["loss"] for v in self.evaluator.get_validation_loss("LSTM")]
        val_g = [v["loss"] for v in self.evaluator.get_validation_loss("GRU")]
        val_s = [v["loss"] for v in self.evaluator.get_validation_loss("STGNN")]

        self.evaluator.plot_loss(
            hist_l=self.evaluator.get_training_loss("LSTM"),
            hist_g=self.evaluator.get_training_loss("GRU"),
            hist_s=self.evaluator.get_training_loss("STGNN"),
            val_l=val_l,
            val_g=val_g,
            val_s=val_s
        )

        return EvaluationResult(
            y_true=y_true_all,
            y_pred=y_pred_all,
            probs=probs_all,
            prediction_dates=prediction_dates,
            hist_train=self.evaluator.get_training_loss(self.model_name),
            hist_val=self.evaluator.get_validation_loss(self.model_name),
            horizon=self.prediction_horizon,
            model_name=self.model_name
        )

    def _unpack_batch(self, batch):
        if isinstance(batch, GeoBatch):
            if batch.x.dim() != 3:
                raise ValueError(f"[Trainer:_unpack_batch] Expected batch.x dim=3, got shape={tuple(batch.x.shape)}")

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

        elif isinstance(batch, (tuple, list)):
            x, y = batch
            return x.to(self.device), y.to(self.device), None, None

        raise ValueError(f"[Trainer:_unpack_batch] Unexpected batch type: {type(batch)}")

    def _forward_and_loss(self, x, y, edge_index, edge_attr=None):
        if self.graphBuilder:
            assert self.targetIdx is not None, "STGNN requires a valid target index"
            pred = self.model(x, edge_index=edge_index, edge_attr=edge_attr, target_node_index=self.targetIdx)
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
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print(f"[EnergyMonitor] Failed to init NVML: {e}")
            self.handle = None

    def _get_gpu_power_usage(self):
        if self.handle:
            try:
                return nvmlDeviceGetPowerUsage(self.handle) / 1000
            except Exception:
                return 0.0
        return 0.0

    def _finalise_energy_monitoring(self):
        try:
            if getattr(self, "handle", None) is not None:
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