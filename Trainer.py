import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as GeoBatch, Data
import pandas as pd
from evaluation_types import EvaluationResult
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown

class Trainer:
    """
    Handles training and evaluation for both LSTM and STGNN models.
    Considers "graph_builder" to differ between LSTM and STGNN (STGNN requires graph_builder).
    """
    def __init__(self, model, optimiser, criterion, device,
                 graphBuilder=None, features=None,
                 tickers=None, targetTicker=None,
                 frontend=None, evaluator=None,
                 prediction_horizon = None,
                 seq_len = None,
                 model_name = None, weight_decay=None
                 ):
		# === STEP 1: Configuration Setup ===
        # ------------------------------------

        #   1. Base configuration
        self.frontend = frontend
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.device = device        
        self.evaluator = evaluator

        #   2. Stock configuration
        self.features = features or {}
        self.tickers = tickers or []
        self.prediction_horizon = prediction_horizon
        self.seq_len = seq_len

        #   3. STGNN configuration
        self.graphBuilder = graphBuilder
        self.targetTicker = targetTicker
        if self.graphBuilder is not None and hasattr(self.graphBuilder, 'edge_index'):
            self.edge_index = self.graphBuilder.edge_index.to(self.device)
        else:
            self.edge_index = None
        self.targetIdx = self._resolve_target_index()
        self.model.edge_index = self.edge_index
        self.model.target_node_index = self.targetIdx
        self.model_name = model_name

        #   4. Weight decay & gradient clipping
        self.weight_decay = float(weight_decay) if weight_decay is not None else 0.0
        self.clip_max_norm = 1.0          # default on; set to None or <=0 to disable
        self._clip_activations = 0        # counter for diagnostics
        self.decision_threshold = 0.5

    def train(self, dataloader, num_epochs, stop_event=None):
        # === STEP 1: Start Memory Tracking ===
        # ------------------------------------
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

        if self.weight_decay and self.weight_decay > 0:
            self._apply_weight_decay(self.weight_decay, exclude_norm_bias=True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            factor=0.5,
            patience=5
        )
        
		# === STEP 2: Training Loop ===
        # ------------------------------------
        for epoch in range(num_epochs):
            #   1. Training mode
            epoch_samples = 0
            self.model.train()
            epoch_losses = []
            epoch_start_time = time.time()
            power_samples = []

            #   2. Sample initial power reading
            power_samples = []
            power_checkpoints = 0

            for batch_idx, batch in enumerate(dataloader):
                try:
                    #   3. Sample power at each batch step
                    power_samples.append(self._get_gpu_power_usage())
                    power_checkpoints += 1

    		        # === STEP 3: Batch Handling ===
            	    # ------------------------------------
                    x, y, edge_index, edge_attr = self._unpack_batch(batch)

                    # Robust batch-size detection
                    batch_size = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1
                    epoch_samples += int(batch_size)

                    # === STEP 4: Compute Loss ===
            	    # ------------------------------------
                    loss = self._forward_and_loss(x, y, edge_index, edge_attr)
                    loss.backward()
                    self._clip_grads()
                    self.optimiser.step()
                    self.optimiser.zero_grad()
                    epoch_losses.append(loss.item())

                    # === STEP 5: Progress Tracking ===
            	    # ------------------------------------
                    progress = ((epoch * len(dataloader)) + batch_idx + 1) / (num_epochs * len(dataloader))
                    if self.frontend is not None:
                        self.frontend.updateProgress(progress)
		            # === STEP 6: Check Termination ===
            	    # ------------------------------------
                    if stop_event and stop_event.is_set():
                        print("[Trainer] Early stopping triggered.")
                        time.sleep(0.01)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        self._finalise_energy_monitoring()
                        return

                except Exception as e:
                    print(f"[Trainer] Skipping batch due to error: {e}")
                    continue

            # === STEP 7: Epoch Statistics ===
            # ------------------------------------
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

            # === STEP 8: Live Tracking ===
            # ------------------------------------

            #   1. Set model to plot
            model_key = self.model_name

            #   2. Store loss
            self.evaluator.record_training_loss(model_key, avg_loss)

            #   3. Plot loss
            self.evaluator.plot_loss(
                hist_l=self.evaluator.get_training_loss("LSTM"),
                hist_s=self.evaluator.get_training_loss("STGNN"),
                hist_g=self.evaluator.get_training_loss("GRU"),
                val_l=[v["loss"] for v in self.evaluator.get_validation_loss("LSTM")],
                val_s=[v["loss"] for v in self.evaluator.get_validation_loss("STGNN")],
                val_g=[v["loss"] for v in self.evaluator.get_validation_loss("GRU")],
            )

		# === STEP 9: Terminate Memory Tracking ===
        # ------------------------------------
        self.total_energy_Wh = total_energy_Wh
        self.total_train_seconds = total_train_seconds
        self.avg_power_W = (total_energy_Wh * 3600.0 / total_train_seconds) if total_train_seconds > 0 else 0.0

        self.total_samples = total_samples
        self.energy_per_sample_Wh = (total_energy_Wh / total_samples) if total_samples > 0 else 0.0

        print(f"[Training Summary] Total energy used: {self.total_energy_Wh:.4f} Wh")
        print(f"[Training Summary] Total energy used: {self.total_energy_Wh:.4f} Wh")
        print(f"[Training Summary] Batches with gradient clipping: {self._clip_activations}")

        self._finalise_energy_monitoring()
        return self.total_energy_Wh  # optional, but convenient
    
    def evaluate_rolling(self, val_dataset):
        # === STEP 1: Evaluation Mode ===
        # ------------------------------------
        self.model.eval()

        # === STEP 2: Evaluation Set-up ===
        # ------------------------------------
        total_samples = len(val_dataset)

        y_true_all, y_pred_all, probs_all, prediction_dates = [], [], [], []
        losses = []

        # === STEP 3: Evaluation Loop (dense) ===
        # ------------------------------------
        with torch.no_grad():
            for i in range(0, total_samples, 1):
                try:
                    # 1) Fetch sample
                    sample = val_dataset[i]
                    timestamp = None

                    # 2) Prepare inputs per model type
                    edge_index = None
                    edge_attr = None

                    # STGNN case: torch_geometric.data.Data
                    if isinstance(sample, Data):
                        x = sample.x.unsqueeze(0).to(self.device)
                        y = sample.y.unsqueeze(0).to(self.device)
                        edge_index = sample.edge_index.to(self.device)
                        if hasattr(sample, "edge_attr") and sample.edge_attr is not None:
                            edge_attr = sample.edge_attr.to(self.device)

                    # LSTM / GRU case: tuple (x, y)
                    elif isinstance(sample, tuple):
                        x, y = sample
                        x = x.unsqueeze(0).to(self.device)
                        y = y.unsqueeze(0).to(self.device)

                    else:
                        print(f"[WARNING] Unrecognised sample format at index={i}, skipping.")
                        continue

                    # 3) Timestamp
                    timestamp = val_dataset.get_timestamp(i)
                    if timestamp:
                        timestamp = pd.Timestamp(timestamp).tz_localize(None)
                        prediction_dates.append(timestamp)
                    else:
                        print(f"[DEBUG] No timestamp found at index={i}")

                    # 4) Forward pass
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

                    # 5) Bookkeeping
                    y_pred_all.append(pred)
                    y_true_all.append(truth)
                    probs_all.append(prob)

                    # 6) Loss (ensure shapes match)
                    if y.shape != logits.shape:
                        y = y.view_as(logits)
                    loss = self.criterion(logits, y)
                    losses.append(loss.item())

                    # 7) Record point loss for plotting over time
                    self.evaluator.record_validation_loss(self.model_name, loss.item(), timestamp)

                except Exception as e:
                    print(f"[ERROR] Skipping sample index={i} due to: {e}")
                    continue

        # === STEP 4: Summary ===
        # ------------------------------------
        mean_val_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[EVAL][SUMMARY] total_samples={total_samples}, predictions={len(prediction_dates)}")
        print(f"[EVAL] Predicted 1s: {sum(y_pred_all)}, 0s: {len(y_pred_all) - sum(y_pred_all)}")
        print(f"[EVAL] True 1s: {sum(y_true_all)}, 0s: {len(y_true_all) - sum(y_true_all)}")
        print(f"[EVAL] Mean validation loss (dense): {mean_val_loss:.6f}")

        # === STEP 5: Plotting (train vs val for all models) ===
        # ------------------------------------
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

        # === STEP 6: Return evaluation artefacts ===
        # ------------------------------------
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
            x = batch.x.view(batch.num_graphs, len(self.tickers), batch.x.size(1), batch.x.size(2))
            y = batch.y.view(batch.num_graphs, -1)
            edge_index = batch.edge_index
            edge_attr  = getattr(batch, 'edge_attr', None)

            # — safety: dtype + shape
            edge_index = edge_index.to(self.device)
            if edge_index.dtype != torch.long:
                edge_index = edge_index.to(torch.long)
            if edge_index.dim() == 1:
                assert edge_index.numel() % 2 == 0, "edge_index 1-D length must be even"
                edge_index = edge_index.view(2, -1).contiguous()
            elif edge_index.dim() == 2 and edge_index.size(0) != 2:
                edge_index = edge_index.t().contiguous()

            edge_attr = (edge_attr.to(self.device) if edge_attr is not None else None)
            return x.to(self.device), y.to(self.device), edge_index, edge_attr

        # === STEP 2: Non-GeoBatch Handling (LSTM) ===
        # ------------------------------------
        elif isinstance(batch, (tuple, list)):
            x, y = batch
            return x.to(self.device), y.to(self.device), None, None

        raise ValueError(f"[Trainer:_unpack_batch] Unexpected batch type or size: {type(batch)}, {len(batch)}")
    
    def _forward_and_loss(self, x, y, edge_index, edge_attr=None):
        # ====================================
		# === Helper to compute loss
        if self.graphBuilder:
            assert self.targetIdx is not None, "STGNN requires a valid target index"
            pred = self.model(x, edge_index=edge_index, edge_attr=edge_attr, target_node_index=self.targetIdx)
        else:
            pred = self.model(x)

        y = y.view(-1, 1)
        loss = self.criterion(pred, y)
        return loss
    
    def _resolve_target_index(self):
        # ====================================
		# === Helper to predict for a single stock (STGNN)
        if self.graphBuilder and self.targetTicker in self.tickers:
            return self.tickers.index(self.targetTicker)
        return None
    
    def _init_energy_monitoring(self):
        # ====================================
		# === Helper to initialise energy monitoring
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)  # assumes 1 GPU
        except Exception as e:
            print(f"[EnergyMonitor] Failed to init NVML: {e}")
            self.handle = None

    def _get_gpu_power_usage(self):
        # ====================================
		# === Helper to get GPU power usage
        if self.handle:
            try:
                power = nvmlDeviceGetPowerUsage(self.handle) / 1000  # milliwatts to watts
                return power
            except:
                return 0.0
        return 0.0

    def _finalise_energy_monitoring(self):
        # ====================================
		# === Helper to terminate energy tracking
        nvmlShutdown()

    def _clip_grads(self):
        """Clip parameter gradients by global norm; return unclipped total norm."""
        if not self.clip_max_norm or self.clip_max_norm <= 0:
            return 0.0
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
        # Count only when clipping actually engaged
        try:
            engaged = float(total_norm) > float(self.clip_max_norm)
        except Exception:
            engaged = False
        if engaged:
            self._clip_activations += 1
        return float(total_norm)
