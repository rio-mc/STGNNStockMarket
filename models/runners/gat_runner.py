from architectures import GATGraphClassifier
from core.utils.utils import Utils

from ..base_runner import BaseModelRunner, ModelRunResult


class GATRunner(BaseModelRunner):
    model_name = "GAT"

    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        app.frontendApp.set_status("Training GAT baseline...")
        self._prepare_memory_logging(self.model_name)

        train_ds, val_ds, test_ds, aligned_tickers, num_nodes = self._build_graph_datasets(app, stock)
        self._log_dataset_summary(app, train_ds, val_ds, test_ds, aligned_tickers)

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

        trainer = self._make_trainer(
            app=app,
            model=model,
            stock=stock,
            evaluator=evaluator,
            lr=app.args.stgnn_lr,
            graph_builder=app.graphBuilder,
            features={"feature": None},
            tickers=aligned_tickers,
            train_dataset=train_ds,
        )
        self._set_target_from_dataset(trainer, model, train_ds)

        dl = self._make_geo_loader(app, train_ds)
        val_dl = self._make_geo_loader(app, val_ds, shuffle=False)
        self._train_model(
            app=app,
            trainer=trainer,
            dataloader=dl,
            validation_dataloader=val_dl,
            epochs=app.args.stgnn_epochs,
            stop_event=stop_event,
        )

        result = self._evaluate_and_predict(
            app=app,
            stock=stock,
            price_df=price_df,
            evaluator=evaluator,
            stop_event=stop_event,
            model=model,
            trainer=trainer,
            val_ds=val_ds,
            test_ds=test_ds,
            live_predict_fn=lambda: self._live_graph_probability(app, model, test_ds),
            eval_status="Evaluating GAT baseline...",
            predict_status="Predicting with GAT baseline...",
        )

        self._cleanup(dl, val_dl, train_ds, val_ds, test_ds)
        return result
