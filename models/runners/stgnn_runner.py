import gc
import time
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataLoader as GeoDataLoader

from training.trainer import Trainer
from data import STGNNDataset
from architectures import STGNNClassifier
from core.utils.utils import Utils

from ..base_runner import BaseModelRunner, ModelRunResult


class STGNNRunner(BaseModelRunner):
    model_name = "STGNN"

    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        app.frontendApp.set_status("Training STGNN...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        Utils.log_gpu_memory("Before STGNN")

        stgnn_train_ds = STGNNDataset(
            graph_builder=app.graphBuilder,
            feature_dict=app.train_feats,
            tickers=app.args.tickers,
            edge_index=app.init_edge_index,
            target_ticker=stock,
            feature_cols=app.raw_feature_cols,
            seq_len=app.seq_len,
            horizon=app.horizon,
            include_target_flag=True,
        )

        stgnn_val_ds = STGNNDataset(
            graph_builder=app.graphBuilder,
            feature_dict=app.val_feats,
            tickers=app.args.tickers,
            edge_index=app.init_edge_index,
            target_ticker=stock,
            feature_cols=app.raw_feature_cols,
            seq_len=app.seq_len,
            horizon=app.horizon,
            include_target_flag=True,
        )

        aligned_tickers = stgnn_train_ds.tickers
        num_nodes = len(aligned_tickers)

        app.logger.info("[STGNNRunner] train_ds size=%d", len(stgnn_train_ds))
        app.logger.info("[STGNNRunner] val_ds size=%d", len(stgnn_val_ds))
        app.logger.info("[STGNNRunner] aligned_tickers=%s", aligned_tickers)

        if len(stgnn_train_ds) == 0:
            raise RuntimeError("STGNN training dataset is empty.")
        if len(stgnn_val_ds) == 0:
            raise RuntimeError("STGNN validation dataset is empty.")

        model = STGNNClassifier(
            edge_index=app.init_edge_index,
            num_nodes=num_nodes,
            feature_dim=len(app.raw_feature_cols) + 1,
            tcn_channels=app.args.tcn_channels,
            tcn_kernel=app.args.tcn_kernel_size,
            gcn_hidden=app.args.gcn_hidden,
            stgnn_blocks=app.args.stgnn_blocks,
            out_dim=1,
            dropout=app.args.dropout,
            rep_dim=app.args.rep_dim,
            head_hidden=app.args.head_hidden,
            graph_model=getattr(app.args, "graph_model", "gcn"),
        ).to(app.device)

        params = Utils.count_parameters(model)
        app.logger.info("STGNN parameters: %s", f"{params:,}")

        mode = getattr(app.args, "graph_ablation", "none")

        if mode == "empty":
            edge_attr = None
        elif mode == "identity":
            edge_attr = torch.ones((num_nodes, 1), dtype=torch.float32)
        else:
            edge_attr = app.graphBuilder.build_edge_weight_tensor(app.init_edge_index)

        model.edge_attr = edge_attr.to(app.device) if edge_attr is not None else None

        sample0 = stgnn_train_ds[0]
        app.logger.info("[STGNNRunner] sample0 type=%s", type(sample0))
        if hasattr(sample0, "x"):
            app.logger.info("[STGNNRunner] sample0.x shape=%s", tuple(sample0.x.shape))
        if hasattr(sample0, "y"):
            try:
                app.logger.info("[STGNNRunner] sample0.y shape=%s", tuple(sample0.y.shape))
            except Exception:
                app.logger.info("[STGNNRunner] sample0.y=%s", sample0.y)
        if hasattr(sample0, "edge_index"):
            app.logger.info("[STGNNRunner] sample0.edge_index shape=%s", tuple(sample0.edge_index.shape))
        if hasattr(sample0, "edge_attr") and sample0.edge_attr is not None:
            app.logger.info("[STGNNRunner] sample0.edge_attr shape=%s", tuple(sample0.edge_attr.shape))

        dl_stgnn_train = GeoDataLoader(
            stgnn_train_ds,
            batch_size=app.args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=app.dl_gen,
            worker_init_fn=app._seed_worker,
        )

        first_batch = next(iter(dl_stgnn_train))
        app.logger.info("[STGNNRunner] first batch type=%s", type(first_batch))
        if hasattr(first_batch, "x"):
            app.logger.info("[STGNNRunner] first batch x shape=%s", tuple(first_batch.x.shape))
        if hasattr(first_batch, "y"):
            app.logger.info("[STGNNRunner] first batch y shape=%s", tuple(first_batch.y.shape))
        if hasattr(first_batch, "edge_index"):
            app.logger.info("[STGNNRunner] first batch edge_index shape=%s", tuple(first_batch.edge_index.shape))
        if hasattr(first_batch, "edge_attr") and first_batch.edge_attr is not None:
            app.logger.info("[STGNNRunner] first batch edge_attr shape=%s", tuple(first_batch.edge_attr.shape))

        trainer = Trainer(
            model,
            Utils.make_adamw(model, lr=app.args.stgnn_lr, weight_decay=app.args.weight_decay),
            BCEWithLogitsLoss(),
            app.device,
            app.graphBuilder,
            {"feature": None},
            aligned_tickers,
            stock,
            app.frontendApp,
            evaluator,
            prediction_horizon=app.horizon,
            seq_len=app.seq_len,
            model_name=self.model_name,
        )
        trainer.targetIdx = stgnn_train_ds.target_idx
        model.target_node_index = stgnn_train_ds.target_idx
        
        start = time.time()
        trainer.train(
            dl_stgnn_train,
            num_epochs=app.args.stgnn_epochs,
            stop_event=stop_event,
            patience=app.args.early_stopping_patience,
        )
        app.logger.info("[STGNNRunner] Training completed in %.2fs", time.time() - start)
        Utils.log_gpu_memory("After STGNN")

        app._check_stop(stop_event)

        app.frontendApp.set_status("Evaluating STGNN...")
        eval_result = trainer.evaluate_rolling(stgnn_val_ds)
        self._attach_metadata(app, eval_result)
        
        if hasattr(model, "classifier") and hasattr(model.classifier, "set_temperature"):
            model.classifier.set_temperature(app.args.head_temperature)

        metrics = evaluator.evaluate(
            model_name=self.model_name,
            result=eval_result,
            price_df=price_df,
        )

        app._check_stop(stop_event)

        app.frontendApp.set_status("Predicting with STGNN...")

        live_graph = stgnn_val_ds[len(stgnn_val_ds) - 1]
        x_live = live_graph.x.unsqueeze(0).to(app.device)
        edge_index_live = live_graph.edge_index.to(app.device)
        edge_attr_live = (
            live_graph.edge_attr.to(app.device)
            if getattr(live_graph, "edge_attr", None) is not None
            else None
        )

        with torch.no_grad():
            logits = model(
                x_live,
                edge_index=edge_index_live,
                edge_attr=edge_attr_live,
                target_node_index=stgnn_val_ds.target_idx,
            )
            prob = torch.sigmoid(logits.view(-1)[0]).item()

        threshold = self._resolve_threshold(
            metrics,
            policy=getattr(app.args, "decision_threshold_policy", "fixed"),
        )

        if prob >= threshold:
            direction = "Upwards"
            confidence = prob * 100.0
        else:
            direction = "Downwards"
            confidence = (1.0 - prob) * 100.0

        app.logger.info("[STGNNRunner] %s (%.1f%%)", direction, confidence)

        result = ModelRunResult(
            model_name=self.model_name,
            direction=direction,
            confidence=confidence,
            metrics=metrics,
            eval_result=eval_result,
            trainer=trainer,
            model=model,
        )

        del first_batch, dl_stgnn_train, stgnn_train_ds, stgnn_val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result