import gc
import time
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataLoader as GeoDataLoader

from trainer.trainer import Trainer
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

        model = STGNNClassifier(
            edge_index=app.init_edge_index,
            num_nodes=len(app.args.tickers),
            feature_dim=len(app.raw_feature_cols) + 1,
            tcn_channels=app.args.tcn_channels,
            tcn_kernel=app.args.tcn_kernel_size,
            gcn_hidden=app.args.gcn_hidden,
            stgnn_blocks=app.args.stgnn_blocks,
            out_dim=1,
            dropout=app.args.dropout,
            rep_dim=app.args.rep_dim,
            head_hidden=app.args.head_hidden,
        ).to(app.device)

        params = Utils.count_parameters(model)
        app.logger.info(f"STGNN parameters: {params:,}")

        mode = getattr(app.args, "graph_ablation", "none")
        num_nodes = len(app.args.tickers)

        if mode == "empty":
            edge_attr = None
        elif mode == "identity":
            edge_attr = torch.ones((num_nodes, 1), dtype=torch.float32)
        else:
            edge_attr = app.graphBuilder.build_edge_weight_tensor(app.init_edge_index)

        model.edge_attr = edge_attr.to(app.device) if edge_attr is not None else None

        stgnn_train_ds = STGNNDataset(
            app.tf_train,
            app.graphBuilder,
            app.train_feats,
            app.args.tickers,
            app.init_edge_index,
            stock,
            app.horizon,
        )

        stgnn_val_ds = STGNNDataset(
            app.tf_val,
            app.graphBuilder,
            app.val_feats,
            app.args.tickers,
            app.init_edge_index,
            stock,
            app.horizon,
        )

        app.logger.info(f"[STGNNRunner] train_ds size={len(stgnn_train_ds)}")
        app.logger.info(f"[STGNNRunner] val_ds size={len(stgnn_val_ds)}")

        if len(stgnn_train_ds) == 0:
            raise RuntimeError("STGNN training dataset is empty.")
        if len(stgnn_val_ds) == 0:
            raise RuntimeError("STGNN validation dataset is empty.")

        sample0 = stgnn_train_ds[0]
        app.logger.info(f"[STGNNRunner] sample0 type={type(sample0)}")
        if hasattr(sample0, "x"):
            app.logger.info(f"[STGNNRunner] sample0.x shape={tuple(sample0.x.shape)}")
        if hasattr(sample0, "y"):
            try:
                app.logger.info(f"[STGNNRunner] sample0.y shape={tuple(sample0.y.shape)}")
            except Exception:
                app.logger.info(f"[STGNNRunner] sample0.y={sample0.y}")
        if hasattr(sample0, "edge_index"):
            app.logger.info(f"[STGNNRunner] sample0.edge_index shape={tuple(sample0.edge_index.shape)}")
        if hasattr(sample0, "edge_attr") and sample0.edge_attr is not None:
            app.logger.info(f"[STGNNRunner] sample0.edge_attr shape={tuple(sample0.edge_attr.shape)}")

        dl_stgnn_train = GeoDataLoader(
            stgnn_train_ds,
            batch_size=app.args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=app.dl_gen,
            worker_init_fn=app._seed_worker,
        )

        app.logger.info("[STGNNRunner] Probing first dataloader batch...")
        first_batch = next(iter(dl_stgnn_train))
        app.logger.info(f"[STGNNRunner] first batch type={type(first_batch)}")
        if hasattr(first_batch, "x"):
            app.logger.info(f"[STGNNRunner] first batch x shape={tuple(first_batch.x.shape)}")
        if hasattr(first_batch, "y"):
            app.logger.info(f"[STGNNRunner] first batch y shape={tuple(first_batch.y.shape)}")
        if hasattr(first_batch, "edge_index"):
            app.logger.info(f"[STGNNRunner] first batch edge_index shape={tuple(first_batch.edge_index.shape)}")
        if hasattr(first_batch, "edge_attr") and first_batch.edge_attr is not None:
            app.logger.info(f"[STGNNRunner] first batch edge_attr shape={tuple(first_batch.edge_attr.shape)}")

        trainer = Trainer(
            model,
            Utils.make_adamw(model, lr=app.args.stgnn_lr, weight_decay=app.args.weight_decay),
            BCEWithLogitsLoss(),
            app.device,
            app.graphBuilder,
            {"feature": None},
            app.args.tickers,
            stock,
            app.frontendApp,
            evaluator,
            prediction_horizon=app.horizon,
            seq_len=app.seq_len,
            model_name=self.model_name,
        )

        app.logger.info("[STGNNRunner] Starting training...")
        start = time.time()
        trainer.train(
            dl_stgnn_train,
            num_epochs=app.args.stgnn_epochs,
            stop_event=stop_event,
        )
        app.logger.info(f"[STGNNRunner] Training completed in {time.time() - start:.2f}s")
        Utils.log_gpu_memory("After STGNN")

        app._check_stop(stop_event)

        app.frontendApp.set_status("Evaluating STGNN...")
        eval_result = trainer.evaluate_rolling(stgnn_val_ds)

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
        edge_attr_live = live_graph.edge_attr.to(app.device) if getattr(live_graph, "edge_attr", None) is not None else None

        with torch.no_grad():
            logits = model(
                x_live,
                edge_index=edge_index_live,
                edge_attr=edge_attr_live,
                target_node_index=app.args.tickers.index(stock),
            )
            prob = torch.sigmoid(logits.view(-1)[0]).item()

        threshold = metrics.get("best_threshold", 0.5)
        if prob >= threshold:
            direction = "Upwards"
            confidence = prob * 100.0
        else:
            direction = "Downwards"
            confidence = (1.0 - prob) * 100.0

        app.logger.info(f"[STGNNRunner] {direction} ({confidence:.1f}%)")

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