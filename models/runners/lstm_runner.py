import gc
import time
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

from training.trainer import Trainer
from data import RecurrentDataset
from architectures import LSTMClassifier
from core.utils.utils import Utils

from ..base_runner import BaseModelRunner, ModelRunResult


class LSTMRunner(BaseModelRunner):
    model_name = "LSTM"

    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        app.frontendApp.set_status("Training LSTM...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        Utils.log_gpu_memory("Before LSTM")

        train_df_stock = app.train_feats[stock].iloc[-app.min_train_len:]
        val_df_stock = app.val_feats[stock].iloc[-app.min_val_len:]

        lstm_train_ds = RecurrentDataset(
            app.tf_train,
            {stock: train_df_stock},
            stock,
            app.horizon,
        )

        lstm_val_ds = RecurrentDataset(
            app.tf_val,
            {stock: val_df_stock},
            stock,
            app.horizon,
        )

        dl_lstm_train = torch.utils.data.DataLoader(
            lstm_train_ds,
            batch_size=app.args.batch_size,
            shuffle=True,
            generator=app.dl_gen,
            worker_init_fn=app._seed_worker,
        )

        model = LSTMClassifier(
            feature_dim=len(app.raw_feature_cols),
            hidden_dim=app.args.lstm_hidden,
            num_layers=app.args.lstm_layers,
            out_channels=1,
            bidirectional=app.args.bidirectional,
            dropout=app.args.dropout,
            rep_dim=app.args.rep_dim,
            head_hidden=app.args.head_hidden,
        ).to(app.device)

        params = Utils.count_parameters(model)
        app.logger.info(f"LSTM parameters: {params:,}")

        trainer = Trainer(
            model,
            Utils.make_adamw(model, lr=app.args.lstm_lr, weight_decay=app.args.weight_decay),
            BCEWithLogitsLoss(),
            app.device,
            graphBuilder=None,
            features=None,
            tickers=[stock],
            targetTicker=stock,
            frontend=app.frontendApp,
            evaluator=evaluator,
            prediction_horizon=app.horizon,
            seq_len=app.seq_len,
            model_name=self.model_name,
        )

        start = time.time()
        trainer.train(
            dl_lstm_train,
            app.args.lstm_epochs,
            stop_event=stop_event,
        )
        app.logger.info(f"[LSTMRunner] Training completed in {time.time() - start:.2f}s")
        Utils.log_gpu_memory("After LSTM")

        app._check_stop(stop_event)

        app.frontendApp.set_status("Evaluating LSTM...")
        trainer.prediction_horizon = app.horizon
        eval_result = trainer.evaluate_rolling(lstm_val_ds)

        if hasattr(model, "classifier") and hasattr(model.classifier, "set_temperature"):
            model.classifier.set_temperature(app.args.head_temperature)

        metrics = evaluator.evaluate(
            model_name=self.model_name,
            result=eval_result,
            price_df=price_df,
        )

        app._check_stop(stop_event)

        app.frontendApp.set_status("Predicting with LSTM...")
        live_x = val_df_stock[app.raw_feature_cols].iloc[-app.seq_len:].values.astype(np.float32)
        arr_x = torch.tensor(live_x, dtype=torch.float32, device=app.device).unsqueeze(0)

        with torch.no_grad():
            prob = torch.sigmoid(model(arr_x)[0]).item()

        threshold = metrics.get("best_threshold", 0.5)
        if prob >= threshold:
            direction = "Upwards"
            confidence = prob * 100.0
        else:
            direction = "Downwards"
            confidence = (1.0 - prob) * 100.0

        app.logger.info(f"[LSTMRunner] {direction} ({confidence:.1f}%)")

        result = ModelRunResult(
            model_name=self.model_name,
            direction=direction,
            confidence=confidence,
            metrics=metrics,
            eval_result=eval_result,
            trainer=trainer,
            model=model,
        )

        del dl_lstm_train, lstm_train_ds, lstm_val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result