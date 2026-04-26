import gc
import time

import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataLoader as GeoDataLoader

from training.trainer import Trainer
from data import STGNNDataset
from architectures import GraphSAGEGraphClassifier
from core.utils.utils import Utils

from ..base_runner import BaseModelRunner, ModelRunResult


class GraphSAGERunner(BaseModelRunner):
    model_name = "GRAPHSAGE"

    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        app.frontendApp.set_status("Training GraphSAGE baseline...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        Utils.log_gpu_memory("Before GRAPHSAGE")

        train_ds = STGNNDataset(
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

        val_ds = STGNNDataset(
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

        aligned_tickers = train_ds.tickers
        num_nodes = len(aligned_tickers)

        app.logger.info("[GraphSAGERunner] train_ds size=%d", len(train_ds))
        app.logger.info("[GraphSAGERunner] val_ds size=%d", len(val_ds))
        app.logger.info("[GraphSAGERunner] aligned_tickers=%s", aligned_tickers)

        if len(train_ds) == 0:
            raise RuntimeError("GRAPHSAGE training dataset is empty.")
        if len(val_ds) == 0:
            raise RuntimeError("GRAPHSAGE validation dataset is empty.")

        model = GraphSAGEGraphClassifier(
            edge_index=app.init_edge_index,
            num_nodes=num_nodes,
            feature_dim=len(app.raw_feature_cols) + 1,
            gcn_hidden=app.args.gcn_hidden,
            out_dim=1,
            dropout=app.args.dropout,
            rep_dim=app.args.rep_dim,
            head_hidden=app.args.head_hidden,
        ).to(app.device)

        params = Utils.count_parameters(model)
        app.logger.info("GRAPHSAGE parameters: %s", f"{params:,}")

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

        trainer.targetIdx = train_ds.target_idx
        model.target_node_index = train_ds.target_idx

        dl = GeoDataLoader(
            train_ds,
            batch_size=app.args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=app.dl_gen,
            worker_init_fn=app._seed_worker,
        )

        start = time.time()
        trainer.train(
            dl,
            num_epochs=app.args.stgnn_epochs,
            stop_event=stop_event,
        )
        app.logger.info("[GraphSAGERunner] Training completed in %.2fs", time.time() - start)
        Utils.log_gpu_memory("After GRAPHSAGE")

        app._check_stop(stop_event)

        app.frontendApp.set_status("Evaluating GraphSAGE baseline...")
        eval_result = trainer.evaluate_rolling(val_ds)
        self._attach_metadata(app, eval_result)

        if hasattr(model, "classifier") and hasattr(model.classifier, "set_temperature"):
            model.classifier.set_temperature(app.args.head_temperature)

        metrics = evaluator.evaluate(
            model_name=self.model_name,
            result=eval_result,
            price_df=price_df,
        )

        app._check_stop(stop_event)

        app.frontendApp.set_status("Predicting with GraphSAGE baseline...")

        live_graph = val_ds[len(val_ds) - 1]
        x_live = live_graph.x.unsqueeze(0).to(app.device)
        edge_index_live = live_graph.edge_index.to(app.device)
        edge_attr_live = (
            live_graph.edge_attr.to(app.device)
            if getattr(live_graph, "edge_attr", None) is not None
            else None
        )

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(
                model(
                    x_live,
                    edge_index=edge_index_live,
                    edge_attr=edge_attr_live,
                    target_node_index=val_ds.target_idx,
                ).view(-1)[0]
            ).item()

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

        app.logger.info("[GraphSAGERunner] %s (%.1f%%)", direction, confidence)

        result = ModelRunResult(
            model_name=self.model_name,
            direction=direction,
            confidence=confidence,
            metrics=metrics,
            eval_result=eval_result,
            trainer=trainer,
            model=model,
        )

        del dl, train_ds, val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
