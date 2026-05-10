from architectures import GRUClassifier
from core.utils.utils import Utils

from ..base_runner import BaseModelRunner, ModelRunResult


class GRURunner(BaseModelRunner):
    model_name = "GRU"

    def run(self, app, stock: str, price_df, evaluator, stop_event) -> ModelRunResult:
        app.frontendApp.set_status("Training GRU...")
        self._prepare_memory_logging(self.model_name)

        train_ds, val_ds, _train_df_stock, val_df_stock = self._build_recurrent_datasets(app, stock)
        self._log_dataset_summary(app, train_ds, val_ds, getattr(train_ds, "aligned_tickers", [stock]))

        model = GRUClassifier(
            feature_dim=len(app.raw_feature_cols),
            hidden_dim=app.args.lstm_hidden,
            num_layers=app.args.lstm_layers,
            out_channels=1,
            bidirectional=app.args.bidirectional,
            dropout=app.args.dropout,
            rep_dim=app.args.rep_dim,
            head_hidden=app.args.head_hidden,
        ).to(app.device)
        app.logger.info("GRU parameters: %s", f"{Utils.count_parameters(model):,}")

        trainer = self._make_trainer(
            app=app,
            model=model,
            stock=stock,
            evaluator=evaluator,
            lr=app.args.lstm_lr,
            graph_builder=None,
            features=None,
            tickers=[stock],
        )

        dl = self._make_torch_loader(app, train_ds)
        self._train_model(app=app, trainer=trainer, dataloader=dl, epochs=app.args.lstm_epochs, stop_event=stop_event)

        result = self._evaluate_and_predict(
            app=app,
            stock=stock,
            price_df=price_df,
            evaluator=evaluator,
            stop_event=stop_event,
            model=model,
            trainer=trainer,
            val_ds=val_ds,
            live_predict_fn=lambda: self._live_recurrent_probability(app, model, val_df_stock),
            eval_status="Evaluating GRU...",
            predict_status="Predicting with GRU...",
        )

        self._cleanup(dl, train_ds, val_ds)
        return result
