# R5 Error Analysis

```mermaid
flowchart TD
    OUT["EvaluationOutcome<br/>metrics · predictions · gates"]
    META["sample_metadata<br/>village/district/season/year"]

    OUT --> EA["ErrorAnalysis.analyze(outcome, metadata)"]
    META --> EA

    EA --> CC["crop<br/>per-class error rates · top confusions<br/>misclassified samples"]
    EA --> YY["yield<br/>bias · outliers · failures · worst predictions"]
    EA --> GR["group_breakdown<br/>error rate per group"]
    EA --> FG["fusion_analysis<br/>gate means overall/correct/error"]

    CC --> ERR["ErrorAnalysisReport"]
    YY --> ERR
    GR --> ERR
    FG --> ERR

    ERR --> REP["generate_error_analysis_reports"]
    REP --> MD["error_analysis.md/.json"]
    REP --> PNG["fusion_gates.png<br/>image_gate · tabular_gate · fusion_gate<br/>correct vs error buckets"]

    FG --> EXPLAIN["interpretation<br/>high image_gate on errors ⇒ imagery is the weak leg<br/>low fusion_gate on errors ⇒ under-using fusion"]

    style EA fill:#e8f5e9
    style REP fill:#e3f2fd
    style EXPLAIN fill:#ffebee
```

`sample_metadata` must have exactly one entry per evaluated sample or
`ErrorAnalysisError` is raised. The fusion-gate figure is emitted whenever the
evaluator collected per-sample gates.
