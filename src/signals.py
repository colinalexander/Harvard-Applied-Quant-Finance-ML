"""Signal diagnostics helpers (IC series + rolling IC + summary table)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalDiagnostics:
    """Artifacts for Section 6."""

    ic_series: pd.DataFrame  # columns: Date + one column per model name
    ic_rolling: pd.DataFrame  # columns: Date + <model>_roll columns
    summary_table: pd.DataFrame  # columns: model, mean_ic


def compute_signal_diagnostics(
    predictions: pd.DataFrame,
    model_cols: Mapping[str, str],
    *,
    date_col: str = "Date",
    window: int = 126,
    min_periods: int = 20,
) -> SignalDiagnostics:
    """Compute IC series, rolling IC, and mean IC summary (fast path).

    Speed notes:
      - Iterates `groupby` once (no get_group calls).
      - Spearman computed as Pearson correlation on within-date ranks.
      - Avoids copying full predictions: slices required columns first.
    """
    # --- Validate schema (fail fast) ---
    if date_col not in predictions:
        raise KeyError(
            f"compute_signal_diagnostics: missing '{date_col}' in predictions."
        )
    if "Actual" not in predictions:
        raise KeyError("compute_signal_diagnostics: missing 'Actual' in predictions.")

    pred_missing = [c for c in model_cols.values() if c not in predictions]
    if pred_missing:
        raise KeyError(
            f"compute_signal_diagnostics: missing prediction columns: {pred_missing}. "
            f"Available columns: {list(predictions)}"
        )

    # Keep only what we need (reduces payload + avoids full copy)
    use_cols = [date_col, "Actual", *model_cols.values()]
    df = predictions[use_cols].sort_values(date_col)
    df[date_col] = pd.to_datetime(df[date_col])

    # --- Fast Spearman IC helper: Pearson corr on ranks ---
    def _spearman_fast(a: pd.Series, b: pd.Series) -> float:
        m = a.notna() & b.notna()
        if m.sum() < 2:
            return 0.0

        ar = a[m].rank(method="average").to_numpy(dtype=float)
        br = b[m].rank(method="average").to_numpy(dtype=float)

        ar -= ar.mean()
        br -= br.mean()
        denom = np.sqrt((ar * ar).sum() * (br * br).sum())
        if denom == 0.0:
            return 0.0
        return float((ar * br).sum() / denom)

    records: list[dict] = []
    for d, g in df.groupby(date_col, sort=True):
        actual = g["Actual"]
        row = {"Date": d}
        for model_name, pred_col in model_cols.items():
            row[model_name] = _spearman_fast(actual, g[pred_col])
        records.append(row)

    ic_series = (
        pd.DataFrame.from_records(records).sort_values("Date").reset_index(drop=True)
    )

    ic_rolling = ic_series.copy()
    for model_name in model_cols.keys():
        ic_rolling[f"{model_name}_roll"] = (
            ic_series[model_name].rolling(window, min_periods=min_periods).mean()
        )

    summary_table = (
        ic_series.drop(columns=["Date"])
        .mean()
        .rename("mean_ic")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "model"})
        .sort_values("mean_ic", ascending=False)
        .reset_index(drop=True)
    )

    return SignalDiagnostics(
        ic_series=ic_series,
        ic_rolling=ic_rolling,
        summary_table=summary_table,
    )
