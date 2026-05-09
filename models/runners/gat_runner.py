import time

import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataLoader as GeoDataLoader

from architectures import GATGraphClassifier
from core.utils.utils import Utils
from data import STGNNDataset
from training.trainer import Trainer

from ..base_runner import BaseModelRunner, ModelRunResult


class GATRunner(BaseModelRunner):
    model_name = "GAT"

    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        app.frontendApp.set_status("Training GAT baseline...")
        self._prepare_memory_logging(self.model_name)

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

        app.logger.info("[GATRunner] train_ds size=%d", len(train_ds))
        app.logger.info("[GATRunner] val_ds size=%d", len(val_ds))
        app.logger.info("[GATRunner] aligned_tickers=%s", aligned_tickers)

        if len(train_ds) == 0:
            raise RuntimeError("GAT training dataset is empty.")
        if len(val_ds) == 0:
            raise RuntimeError("GAT validation dataset is empty.")

        model = GATGraphClassifier(
            edge_index=app.init_edge_index,
            num_nodes=num_nodes,
            feature_dim=len(app.raw_feature_cols) + 1,
            gcn_hidden=app.args.gcn_hidden,
            out_dim=1,
            dropout=app.args.dropout,
            rep_dim=app.args.rep_dim,
            head_hidden=app.args.head_hidden,
        ).to(app.device)

        app.logger.info("GAT parameters: %s", f"{Utils.count_parameters(model):,}")

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
            patience=app.args.early_stopping_patience,
        )
        app.logger.info("[GATRunner] Training completed in %.2fs", time.time() - start)
        self._log_after_memory(self.model_name)

        def live_predict():
            live_graph = val_ds[len(val_ds) - 1]
            x_live = live_graph.x.unsqueeze(0).to(app.device)
            edge_index_live = live_graph.edge_index.to(app.device)
            edge_attr_live = (
                live_graph.edge_attr.to(app.device)
                if getattr(live_graph, "edge_attr", None) is not None
                else None
            )
            return torch.sigmoid(
                model(
                    x_live,
                    edge_index=edge_index_live,
                    edge_attr=edge_attr_live,
                    target_node_index=val_ds.target_idx,
                ).view(-1)[0]
            ).item()

        result = self._evaluate_and_predict(
            app=app,
            stock=stock,
            price_df=price_df,
            evaluator=evaluator,
            stop_event=stop_event,
            model=model,
            trainer=trainer,
            val_ds=val_ds,
            live_predict_fn=live_predict,
            eval_status="Evaluating GAT baseline...",
            predict_status="Predicting with GAT baseline...",
        )

        self._cleanup(dl, train_ds, val_ds)
        return result