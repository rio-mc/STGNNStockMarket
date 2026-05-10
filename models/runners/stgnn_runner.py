import torch

from architectures import STGNNClassifier
from core.utils.utils import Utils

from ..base_runner import BaseModelRunner, ModelRunResult


class STGNNRunner(BaseModelRunner):
    model_name = "STGNN"

    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        app.frontendApp.set_status("Training STGNN...")
        self._prepare_memory_logging(self.model_name)

        train_ds, val_ds, aligned_tickers, num_nodes = self._build_graph_datasets(app, stock)
        self._log_dataset_summary(app, train_ds, val_ds, aligned_tickers)

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
        app.logger.info("STGNN parameters: %s", f"{Utils.count_parameters(model):,}")

        mode = getattr(app.args, "graph_ablation", "none")
        if mode == "empty":
            edge_attr = None
        elif mode == "identity":
            edge_attr = torch.ones((num_nodes, 1), dtype=torch.float32)
        else:
            edge_attr = app.graphBuilder.build_edge_weight_tensor(app.init_edge_index)
        model.edge_attr = edge_attr.to(app.device) if edge_attr is not None else None

        trainer = self._make_trainer(
            app=app,
            model=model,
            stock=stock,
            evaluator=evaluator,
            lr=app.args.stgnn_lr,
            graph_builder=app.graphBuilder,
            features={"feature": None},
            tickers=aligned_tickers,
        )
        self._set_target_from_dataset(trainer, model, train_ds)

        dl = self._make_geo_loader(app, train_ds)
        self._train_model(app=app, trainer=trainer, dataloader=dl, epochs=app.args.stgnn_epochs, stop_event=stop_event)

        result = self._evaluate_and_predict(
            app=app,
            stock=stock,
            price_df=price_df,
            evaluator=evaluator,
            stop_event=stop_event,
            model=model,
            trainer=trainer,
            val_ds=val_ds,
            live_predict_fn=lambda: self._live_graph_probability(app, model, val_ds),
            eval_status="Evaluating STGNN...",
            predict_status="Predicting with STGNN...",
        )

        self._cleanup(dl, train_ds, val_ds)
        return result
