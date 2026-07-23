from architectures.panel_gru_classifier import PanelGRUClassifier
from core.utils.utils import Utils

from ..base_runner import BaseModelRunner, ModelRunResult


class PanelGRURunner(BaseModelRunner):
    model_name = "PANEL_GRU"

    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        app.frontendApp.set_status("Training PANEL GRU...")
        self._prepare_memory_logging(self.model_name)

        train_ds, val_ds, test_ds, aligned_tickers, _num_nodes = self._build_graph_datasets(app, stock)
        self._log_dataset_summary(app, train_ds, val_ds, test_ds, aligned_tickers)

        model = PanelGRUClassifier(
            feature_dim=len(app.raw_feature_cols) + 1,
            hidden_dim=app.args.lstm_hidden,
            num_layers=app.args.lstm_layers,
            dropout=app.args.dropout,
            rep_dim=app.args.rep_dim,
            head_hidden=app.args.head_hidden,
        ).to(app.device)
        app.logger.info("PANEL_GRU parameters: %s", f"{Utils.count_parameters(model):,}")

        trainer = self._make_trainer(
            app=app,
            model=model,
            stock=stock,
            evaluator=evaluator,
            lr=app.args.lstm_lr,
            graph_builder=None,
            features=None,
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
            epochs=app.args.lstm_epochs,
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
            live_predict_fn=lambda: self._live_panel_probability(app, model, test_ds, pass_target_index=False),
            eval_status="Evaluating PANEL GRU...",
            predict_status="Predicting with PANEL GRU...",
        )

        self._cleanup(dl, val_dl, train_ds, val_ds, test_ds)
        return result
