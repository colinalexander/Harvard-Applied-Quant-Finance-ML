"""Portfolio implementation helpers for Section 7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from src.portfolio import (
    backtest_long_short_forward,
    generate_diagnostics_table,
)


@dataclass
class PortfolioArtifacts:
    """Artifacts produced from `run_portfolio_implementations`."""

    ls_results_by_model: dict[str, pd.DataFrame]
    diagnostics_table: pd.DataFrame


def run_portfolio_implementations(
    *,
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    model_cols: Mapping[str, str],
    rebalance: str = "M",
    top_frac: float = 0.10,
    cost_bps: float = 10.0,
    rf: float = 0.02,
) -> PortfolioArtifacts:
    """Run long–short implementations for a set of model signals.

    Args:
        predictions: DataFrame with Date, PortfolioPair, Actual, and prediction columns.
        returns: DataFrame with Date, PortfolioPair, Return columns (daily).
        model_cols: Mapping {display_name -> prediction_column}.
        rebalance: Rebalance frequency ('M' monthly).
        top_frac: Fraction in long and short legs (e.g., 0.10 = top/bottom decile).
        cost_bps: Transaction cost applied to turnover (10 bps).
        rf: Annual risk-free rate used for Sharpe in diagnostics table.

    Returns:
        PortfolioArtifacts with per-model backtest results and a diagnostics table.
    """
    # Minimal schema checks
    for c in ("Date", "PortfolioPair"):
        if c not in predictions.columns:
            raise KeyError(f"predictions missing required column '{c}'")
    if "Return" not in returns.columns:
        raise KeyError("returns missing required column 'Return'")

    ls_results: dict[str, pd.DataFrame] = {}
    for model_name, pred_col in model_cols.items():
        ls_results[model_name] = backtest_long_short_forward(
            predictions=predictions,
            model_col=pred_col,
            rebalance=rebalance,
            top_frac=top_frac,
            cost_bps=cost_bps,
            ret_col="Actual",
        )

    diagnostics = generate_diagnostics_table(ls_results, rf=rf, periods_per_year=12)
    return PortfolioArtifacts(
        ls_results_by_model=ls_results, diagnostics_table=diagnostics
    )
