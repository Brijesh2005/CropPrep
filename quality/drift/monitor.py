"""DriftMonitor — orchestrates every analyzer into a single report."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .config import DriftConfig
from .feature_drift import FeatureDriftAnalyzer
from .label_drift import LabelDriftAnalyzer
from .prediction_drift import PredictionDriftAnalyzer
from .result import DriftReport, rank_severity as _rank
from .spatial_drift import SpatialDriftAnalyzer
from .temporal_drift import TemporalDriftAnalyzer


class DriftMonitor:
    """Run the full drift battery over a reference vs current dataset.

    Typical usage::

        monitor = DriftMonitor(reference_df, feature_columns=[...], label_column="crop_label")
        report = monitor.evaluate(current_df, predictions=prob_matrix,
                                  lon_lat=pairs, timestamp_column="created_at")
        ReportWriter().write(report, out_dir)
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        *,
        config: DriftConfig | None = None,
        feature_columns: Sequence[str] | None = None,
        label_column: str | None = None,
        prediction_reference: Any = None,
        spatial_reference: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        self.config = config or DriftConfig()
        self.reference = reference.reset_index(drop=True)
        self.feature_columns = list(feature_columns) if feature_columns else None
        self.label_column = label_column
        self.prediction_reference = prediction_reference
        self.spatial_reference = spatial_reference
        self.feature_analyzer = FeatureDriftAnalyzer(self.config)
        self.label_analyzer = LabelDriftAnalyzer(self.config)
        self.prediction_analyzer = PredictionDriftAnalyzer(self.config)
        self.spatial_analyzer = SpatialDriftAnalyzer(self.config)
        self.temporal_analyzer = TemporalDriftAnalyzer(self.config)

    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        current: pd.DataFrame,
        *,
        predictions: Any = None,
        lon_lat: Sequence[tuple[float, float]] | None = None,
        timestamp_column: str | None = None,
        label_column: str | None = None,
        prediction_mode: str = "classification",
    ) -> DriftReport:
        """Evaluate current data and return a :class:`DriftReport`."""
        current = current.reset_index(drop=True)
        label_name = label_column or self.label_column

        features = self.feature_analyzer.analyze(
            self.reference, current, columns=self.feature_columns
        )

        labels = None
        if label_name is not None and label_name in self.reference.columns and label_name in current.columns:
            task = "regression" if pd.api.types.is_numeric_dtype(self.reference[label_name]) else "classification"
            labels = self.label_analyzer.analyze(
                self.reference[label_name], current[label_name], task=task
            )

        predictions_result = None
        if predictions is not None:
            predictions_result = self.prediction_analyzer.analyze(
                self.prediction_reference if self.prediction_reference is not None else predictions,
                predictions,
                mode=prediction_mode,
            )

        spatial = None
        if lon_lat is not None and self.spatial_reference is not None:
            spatial = self.spatial_analyzer.analyze(self.spatial_reference, lon_lat)

        temporal = None
        if timestamp_column is not None and timestamp_column in current.columns:
            temp_column = self.feature_columns[0] if self.feature_columns else (
                current.select_dtypes(include="number").columns[0]
            )
            temporal = self.temporal_analyzer.analyze(
                self.reference[temp_column], current[temp_column], current[timestamp_column]
            )

        report = DriftReport(
            reference_samples=int(len(self.reference)),
            current_samples=int(len(current)),
            features=features,
            labels=labels,
            predictions=predictions_result,
            spatial=spatial,
            temporal=temporal,
        )
        report.overall_severity = _overall_severity(report)
        report.drifted = _overall_drifted(report)
        return report


def _overall_severity(report: DriftReport) -> str:
    severities = [f.severity for f in report.features]
    for result in (report.labels, report.predictions, report.spatial, report.temporal):
        if result is not None:
            severities.append(result.severity)
    if not severities:
        return "low"
    return max(severities, key=_rank)


def _overall_drifted(report: DriftReport) -> bool:
    any_drift = any(f.drifted for f in report.features)
    for result in (report.labels, report.predictions, report.spatial, report.temporal):
        if result is not None and result.drifted:
            any_drift = True
    return any_drift
