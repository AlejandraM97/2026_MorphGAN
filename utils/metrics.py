def format_metrics(metrics_dict):
    for model, metrics in metrics_dict.items():
        print(f"\\nMetrics for '{model}':")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}" if isinstance(value, float) else f"  {metric_name}: {value}")
