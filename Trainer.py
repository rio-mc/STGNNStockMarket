import time
import torch
import torch.nn as nn
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
                 seq_len = None
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
    
    def train(self, dataloader, num_epochs, save_model_path=None, stop_event=None):
        # === STEP 1: Start Memory Tracking ===
        # ------------------------------------
        self._init_energy_monitoring()
        total_energy_Wh = 0.0

		# === STEP 2: Training Loop ===
        # ------------------------------------
        for epoch in range(num_epochs):
            #   1. Training mode
            self.model.train()
            epoch_losses = []
            epoch_start_time = time.time()

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

                    # === STEP 4: Compute Loss ===
            	    # ------------------------------------
                    loss = self._forward_and_loss(x, y, edge_index, edge_attr)
                    loss.backward()
                    self.optimiser.step()
                    self.optimiser.zero_grad()
                    epoch_losses.append(loss.item())

                    # === STEP 5: Progress Tracking ===
            	    # ------------------------------------
                    progress = ((epoch * len(dataloader)) + batch_idx + 1) / (num_epochs * len(dataloader))
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
            avg_power = sum(power_samples) / max(len(power_samples), 1)  # watts
            energy_Wh = avg_power * (epoch_time / 3600)  # watt-hours
            total_energy_Wh += energy_Wh

            print(f"[Epoch {epoch}] Avg Loss = {avg_loss:.4f}")
            print(f"[Epoch {epoch}] Duration = {epoch_time:.2f}s | Avg Power = {avg_power:.2f} W | Energy = {energy_Wh:.4f} Wh")

            # === STEP 8: Live Tracking ===
            # ------------------------------------

            #   1. Set model to plot
            model_key = "STGNN" if self.graphBuilder else "LSTM"

            #   2. Store loss
            self.evaluator.record_training_loss(model_key, avg_loss)

            #   3. Plot loss
            self.evaluator.plot_loss(
                hist_l=self.evaluator.get_training_loss("LSTM"),
                hist_s=self.evaluator.get_training_loss("STGNN"),
                val_l=[v["loss"] for v in self.evaluator.get_validation_loss("LSTM")],
                val_s=[v["loss"] for v in self.evaluator.get_validation_loss("STGNN")],
            )

            if save_model_path:
                torch.save(self.model.state_dict(), save_model_path)

		# === STEP 9: Terminate Memeory Tracking ===
        # ------------------------------------
        print(f"[Training Summary] Total energy used: {total_energy_Wh:.4f} Wh")
        self._finalise_energy_monitoring()
    
    def evaluate_rolling(self, val_dataset):
		# === STEP 1: Evaluation Mode ===
        # ------------------------------------
        self.model.eval()

        # === STEP 2: Evaluation Set-up ===
        # ------------------------------------
        y_true_all, y_pred_all, probs_all = [], [], []
        prediction_dates = []
        total_samples = len(val_dataset)

        # === STEP 3: Evaluation Loop ===
        # ------------------------------------
        for i in range(0, total_samples, self.prediction_horizon):
            try:
                #   1. Collect sample from validation set
                sample = val_dataset[i]
                timestamp = None

                #   2. STGNN evaluation
                if isinstance(sample, Data):
                    x = sample.x.unsqueeze(0).to(self.device)
                    y = sample.y.unsqueeze(0).to(self.device)
                    edge_index = sample.edge_index.to(self.device)
                    edge_attr = getattr(sample, 'edge_attr', None)
                    if edge_attr is not None:
                        edge_attr = edge_attr.to(self.device)

                #   3. LSTM evaluation
                elif isinstance(sample, tuple):
                    x, y = sample
                    x = x.unsqueeze(0).to(self.device)
                    y = y.unsqueeze(0).to(self.device)
                    
                else:
                    print(f"[WARNING] Unrecognised sample format at index={i}, skipping.")
                    continue

                # === STEP 4: Timestamping ===
            	# ------------------------------------

                #   1. Collect timestamp
                timestamp = val_dataset.get_timestamp(i)

                #   2. Format timestamp
                if timestamp:
                    timestamp = pd.Timestamp(timestamp).tz_localize(None)
                    #   3. Add timestamp to prediction dates for chronological plotting
                    prediction_dates.append(timestamp)
                else:
                    print(f"[DEBUG] No timestamp found at index={i}")

                # === STEP 5: Forward Pass ===
            	# ------------------------------------

                #   1. Collect raw output
                logits = (
                    self.model(x, edge_index=edge_index, edge_attr=edge_attr, target_node_index=self.targetIdx)
                    if self.graphBuilder else self.model(x)
                )

                #   2. Sigmoid for normalised output
                prob = torch.sigmoid(logits).detach().cpu().item()
                
                #   3. Establish directional prediction (>=0.5 is upwards)
                pred = int(prob >= 0.5)
                truth = int(y.item())

                #   4. Store predictions and truths for plotting
                y_pred_all.append(pred)
                y_true_all.append(truth)
                probs_all.append(prob)

                #   5. Compute validation loss
                if y.shape != logits.shape:
                    y = y.view_as(logits)
                loss = self.criterion(logits, y)

                #   6. Record losses
                model_key = "STGNN" if self.graphBuilder else "LSTM"
                self.evaluator.record_validation_loss(model_key, loss.item(), timestamp)

            except Exception as e:
                print(f"[ERROR] Skipping sample index={i} due to: {e}")
                continue

        print(f"[EVAL][SUMMARY] {len(prediction_dates)} predictions made.")
        print(f"[EVAL] Predicted 1s: {sum(y_pred_all)}, 0s: {len(y_pred_all) - sum(y_pred_all)}")
        print(f"[EVAL] True 1s: {sum(y_true_all)}, 0s: {len(y_true_all) - sum(y_true_all)}")

		# === STEP 6: Evaluation Plotting ===
        # ------------------------------------

        #   1. Collect model losses
        val_l = [v["loss"] for v in self.evaluator.get_validation_loss("LSTM")]
        val_s = [v["loss"] for v in self.evaluator.get_validation_loss("STGNN")]

        #   2. Plot losses
        self.evaluator.plot_loss(
            hist_l=self.evaluator.get_training_loss("LSTM"),
            hist_s=self.evaluator.get_training_loss("STGNN"),
            val_l=val_l,
            val_s=val_s
        )

        #   3. Store evaluation for modularity
        return EvaluationResult(
            y_true=y_true_all,
            y_pred=y_pred_all,
            probs=probs_all,
            prediction_dates=prediction_dates,
            hist_lstm=self.evaluator.get_training_loss("LSTM") if not self.graphBuilder else None,
            hist_stgnn=self.evaluator.get_training_loss("STGNN") if self.graphBuilder else None,
            val_lstm=self.evaluator.get_validation_loss("LSTM") if not self.graphBuilder else None,
            val_stgnn=self.evaluator.get_validation_loss("STGNN") if self.graphBuilder else None,
            horizon=self.prediction_horizon
        )
    
    def _unpack_batch(self, batch):
        # === STEP 1: GeoBatch Handling (STGNN) ===
        # ------------------------------------
        if isinstance(batch, GeoBatch):
            x = batch.x.view(batch.num_graphs, len(self.tickers), batch.x.size(1), batch.x.size(2))
            y = batch.y.view(batch.num_graphs, -1)
            edge_index = batch.edge_index
            edge_attr = getattr(batch, 'edge_attr', None)
            return x.to(self.device), y.to(self.device), edge_index.to(self.device), (edge_attr.to(self.device) if edge_attr is not None else None)

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