# R5 Evaluation Flow

```mermaid
flowchart TD
    CFG["EvaluationConfig<br/>EVAL_* env > YAML > defaults<br/>device · batch_size · collect_embeddings"]
    LOAD["Phase-4 loader<br/>tabular · ndvi · evi · temporal_mask<br/>crop_label · yield_label"]
    MODEL["Trained CropFusionModel<br/>eval() · no_grad"]

    CFG --> EV["MultimodalEvaluator.evaluate(loader)"]
    LOAD --> EV
    MODEL --> EV

    EV --> MET["Per-task metrics<br/>crop: accuracy · precision · recall · f1 · roc_auc · auprc<br/>yield: rmse · mae · r2 · mape · bias · within_tolerance"]
    EV --> PR["pr_curves (one-vs-rest)"]
    EV --> CM["confusion matrix"]
    EV --> PRED["raw predictions<br/>targets / preds / probs"]
    EV --> EMB["shared embeddings [N,D]"]
    EV --> LAT["latency ms (mean/p50/p95)"]
    EV --> GATE["per-sample gates<br/>image_gate · tabular_gate · fusion_gate"]

    MET --> OUT["EvaluationOutcome"]
    PR --> OUT
    CM --> OUT
    PRED --> OUT
    EMB --> OUT
    LAT --> OUT
    GATE --> OUT

    OUT --> REP["generate_evaluation_reports"]
    REP --> MD["evaluation_report.md/.json<br/>confusion_matrix.png · pr_curves.png<br/>error_histogram.png · per_class_comparison.csv"]

    style EV fill:#e8f5e9
    style REP fill:#e3f2fd
    style OUT fill:#fff3e0
```

A single forward pass per batch produces the predictions, embeddings and gates
together; nothing is re-read from disk. The outcome feeds the comparison,
ablation and error-analysis stages.
