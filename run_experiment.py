from main import MainApp


def main():
    app = MainApp()

    stock = getattr(app.args, "target_stock", None)
    window = getattr(app.args, "prediction_window", None)
    model_name = getattr(app.args, "model", None)

    result = app.run_headless(
        stock=stock,
        gui_window=window,
        model_name=model_name,
    )

    print("\n=== Experiment Result ===")
    print(f"Model      : {result.model_name}")
    print(f"Direction  : {result.direction}")
    print(f"Confidence : {result.confidence:.2f}%")

    if getattr(result, "metrics", None):
        print("Metrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()