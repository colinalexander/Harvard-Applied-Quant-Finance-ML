"""Feature construction for the Harvard applied notebook."""

from __future__ import annotations

import pandas as pd


class FactorPanel(pd.DataFrame):
    """DataFrame subclass with a lightweight summary helper."""

    _metadata = ["_summary_meta"]

    @property
    def _constructor(self):  # pragma: no cover - pandas pattern
        return FactorPanel

    def summary_table(self) -> pd.DataFrame:
        """Return a small summary of rows, columns, and date span."""
        date_col = self["Date"] if "Date" in self.columns else None
        summary = {
            "rows": [len(self)],
            "columns": [len(self.columns)],
        }
        if date_col is not None:
            summary["start"] = [pd.to_datetime(date_col).min()]
            summary["end"] = [pd.to_datetime(date_col).max()]
        return pd.DataFrame(summary)


def build_features(panel: pd.DataFrame) -> FactorPanel:
    """Return the feature panel (placeholder keeps features as-is)."""
    fp = FactorPanel(panel.copy())
    return fp
