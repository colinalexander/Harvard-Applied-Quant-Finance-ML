"""Portfolio construction and backtest helpers for the Harvard notebook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _get_rebalance_dates(dates: pd.Series, freq: str = "M") -> list[pd.Timestamp]:
    """Return sorted rebalance dates using the last available date in each period."""
    dates = pd.to_datetime(dates)
    by_period = pd.DataFrame({"date": dates, "period": dates.dt.to_period(freq)})
    rb_dates = by_period.groupby("period")["date"].max().sort_values().to_list()
    return rb_dates


def backtest_long_short(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    model_col: str,
    rebalance: str = "M",
    top_frac: float = 0.1,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Construct and backtest a long–short portfolio from predictions.

    Args:
        predictions: DataFrame with Date, PortfolioPair, model predictions, and Actual.
        returns: DataFrame with Date, PortfolioPair, Return columns.
        model_col: Name of the prediction column to use.
        rebalance: Pandas offset alias for rebalance frequency (default monthly).
        top_frac: Fraction of names in long and short legs.
        cost_bps: Transaction cost applied to turnover (per-side, round-trip).

    Returns:
        DataFrame with daily gross/net returns, turnover, and counts of
        long/short names.
    """
    preds = predictions.copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    rets = returns.copy()
    rets["Date"] = pd.to_datetime(rets["Date"])

    # Align universe
    common = set(preds["PortfolioPair"].unique()) & set(rets["PortfolioPair"].unique())
    preds = preds[preds["PortfolioPair"].isin(common)]
    rets = rets[rets["PortfolioPair"].isin(common)]

    rb_dates = _get_rebalance_dates(preds["Date"], freq=rebalance)
    rb_weights = {}

    n_names = preds["PortfolioPair"].nunique()
    k = max(1, int(top_frac * n_names))

    for d in rb_dates:
        snap = preds[preds["Date"] == d][["PortfolioPair", model_col]].dropna()
        if snap.empty:
            continue
        ranked = snap.sort_values(model_col)
        short_names = ranked.head(k)["PortfolioPair"].tolist()
        long_names = ranked.tail(k)["PortfolioPair"].tolist()
        weights = {}
        if long_names:
            w = 1.0 / len(long_names)
            for name in long_names:
                weights[name] = w
        if short_names:
            w = -1.0 / len(short_names)
            for name in short_names:
                weights[name] = w
        rb_weights[pd.to_datetime(d)] = weights

    all_dates = sorted(pd.to_datetime(rets["Date"].unique()))

    # Build weight matrix with forward-filled weights
    weight_records = []
    for d, wdict in rb_weights.items():
        for name, w in wdict.items():
            weight_records.append({"Date": d, "PortfolioPair": name, "weight": w})
    weight_df = pd.DataFrame(weight_records)
    if weight_df.empty:
        return pd.DataFrame()

    pivot = weight_df.pivot_table(
        index="Date", columns="PortfolioPair", values="weight", fill_value=np.nan
    )
    pivot = pivot.reindex(all_dates, method="ffill")
    pivot = pivot.fillna(0.0)

    rets_pivot = rets.pivot_table(
        index="Date", columns="PortfolioPair", values="Return", fill_value=0.0
    )
    rets_pivot = rets_pivot.reindex(all_dates, fill_value=0.0)

    # Align columns
    rets_pivot = rets_pivot.reindex(columns=pivot.columns, fill_value=0.0)
    pivot = pivot.reindex(columns=rets_pivot.columns, fill_value=0.0)

    gross_ret = (pivot * rets_pivot).sum(axis=1)

    # Turnover at rebalance dates
    turnover = pd.Series(0.0, index=all_dates)
    prev_w = np.zeros(len(pivot.columns))
    for d in rb_dates:
        if d not in pivot.index:
            continue
        w = pivot.loc[d].to_numpy()
        turnover.loc[d] = 0.5 * np.abs(w - prev_w).sum()
        prev_w = w

    cost = (cost_bps / 10000.0) * turnover
    net_ret = gross_ret - cost

    weight_count_long = (pivot > 0).sum(axis=1)
    weight_count_short = (pivot < 0).sum(axis=1)

    out = pd.DataFrame(
        {
            "date": all_dates,
            "gross_ret": gross_ret.values,
            "net_ret": net_ret.values,
            "turnover": turnover.values,
            "weight_count_long": weight_count_long.values,
            "weight_count_short": weight_count_short.values,
        }
    )
    return out.set_index("date")


def backtest_long_short_forward(
    predictions: pd.DataFrame,
    model_col: str,
    *,
    date_col: str = "Date",
    id_col: str = "PortfolioPair",
    ret_col: str = "Actual",  # <-- forward 21-trading-day return
    rebalance: str = "M",
    top_frac: float = 0.10,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Long–short backtest using forward holding-period returns in `predictions`.

    This is the correct implementation when `ret_col` is a k-trading-day forward return
    (here k=21). The output is one return per rebalance date (not daily).

    Turnover is one-way turnover per rebalance: 0.5 * sum |w_t - w_{t-1}|.
    """
    df = predictions[[date_col, id_col, model_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Universe for turnover accounting
    universe = pd.Index(sorted(df[id_col].dropna().unique()))
    n_universe = len(universe)
    if n_universe == 0:
        return pd.DataFrame()

    rb_dates = _get_rebalance_dates(df[date_col], freq=rebalance)

    prev_w = pd.Series(0.0, index=universe)
    records: list[dict] = []

    for d in rb_dates:
        snap = df[df[date_col] == d].dropna(subset=[model_col, ret_col])
        if snap.empty:
            continue

        # Use available names that day (should be stable at ~100)
        snap = snap.drop_duplicates(subset=[id_col])
        n = snap[id_col].nunique()
        k = max(1, int(top_frac * n))

        ranked = snap.sort_values(model_col, ascending=True)

        short = ranked.head(k)
        long = ranked.tail(k)

        # Equal-weight long/short; gross LS return over the holding period
        gross_ret = float(long[ret_col].mean() - short[ret_col].mean())

        # Turnover (one-way) relative to previous rebalance
        w = pd.Series(0.0, index=universe)
        if len(long) > 0:
            w.loc[long[id_col].values] = 1.0 / len(long)
        if len(short) > 0:
            w.loc[short[id_col].values] = -1.0 / len(short)

        turnover = float(0.5 * (w - prev_w).abs().sum())
        prev_w = w

        cost = (cost_bps / 10000.0) * turnover
        net_ret = gross_ret - cost

        records.append(
            {
                "date": pd.to_datetime(d),
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "turnover": turnover,
                "weight_count_long": int((w > 0).sum()),
                "weight_count_short": int((w < 0).sum()),
            }
        )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    return out.set_index("date").sort_index()


def _max_drawdown(series: pd.Series) -> float:
    """Compute max drawdown from a return series."""
    cumulative = (1 + series).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1.0
    return float(drawdown.min())


def generate_diagnostics_table(
    ls_results_by_model: dict[str, pd.DataFrame],
    *,
    rf: float = 0.02,
    periods_per_year: int = 12,  # <-- 12 for 21-trading-day horizon, 252 for daily
) -> pd.DataFrame:
    """Diagnostics table for long–short backtests.

    Notes:
      - ann_return is geometric annualized return based on realized series.
      - sharpe uses annualized arithmetic mean excess return / annualized vol.
      - avg_turnover is turnover PER REBALANCE (one-way).
      - ann_turnover ≈ avg_turnover * periods_per_year.
    """
    rows = []
    for model, df in ls_results_by_model.items():
        if df is None or df.empty:
            continue

        # turnover here is already per period if df is per rebalance; if df is
        # daily, keep nonzeros
        turn = df["turnover"].dropna()
        turn_nz = turn[turn > 0]
        avg_turn = float(turn_nz.mean()) if not turn_nz.empty else 0.0
        ann_turn = avg_turn * periods_per_year

        for label, col in [("gross", "gross_ret"), ("net", "net_ret")]:
            r = df[col].dropna()
            if r.empty:
                continue

            # Geometric annualized return
            ann_ret = float((1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0)

            # Annualized vol
            ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))

            # Sharpe (annualized arithmetic excess / annualized vol)
            ann_mean = float(r.mean() * periods_per_year)
            sharpe = np.nan if ann_vol == 0 else (ann_mean - rf) / ann_vol

            mdd = _max_drawdown(r)

            rows.append(
                {
                    "model": model,
                    "side": label,
                    "ann_return": ann_ret,
                    "ann_vol": ann_vol,
                    "sharpe": sharpe,
                    "max_drawdown": mdd,
                    "avg_turnover": avg_turn,  # per rebalance
                    "ann_turnover": ann_turn,  # per year (approx)
                }
            )

    table = pd.DataFrame(rows)
    return table


def build_cost_grid_table(
    *,
    predictions: pd.DataFrame,
    model_cols: Mapping[str, str],
    cost_grid_bps: Sequence[float] = (5.0, 10.0, 20.0),
    # portfolio construction controls
    rebalance: str = "M",
    top_frac: float = 0.10,
    # horizon controls (t+21 → ~12 periods/year)
    periods_per_year: int = 12,
    rf: float = 0.02,
    # prediction schema
    date_col: str = "Date",
    id_col: str = "PortfolioPair",
    ret_col: str = "Actual",
    return_results: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[float, dict[str, pd.DataFrame]]]:
    """Build a transaction-cost sensitivity table for long–short implementations.

    This runs identical portfolio construction across a grid of transaction cost
    assumptions and returns a compact table of NET diagnostics (model × cost).

    Args:
        predictions: DataFrame containing Date, PortfolioPair, forward return (ret_col),
            and model prediction columns.
        model_cols: Mapping {model_name -> prediction_column}.
        cost_grid_bps: Transaction cost grid in basis points (e.g., 5, 10, 20).
        rebalance: Rebalance frequency (default monthly).
        top_frac: Fraction in long/short legs.
        periods_per_year: Annualization factor (12 for 21-trading-day horizon).
        rf: Annual risk-free rate for Sharpe calculation.
        date_col/id_col/ret_col: Column names in `predictions`.
        return_results: If True, also return per-cost backtest results dict.

    Returns:
        cost_table: DataFrame with rows = model × cost_bps (net side only),
            columns = ann_return, ann_vol, sharpe, max_drawdown, avg_turnover,
            ann_turnover.
        If return_results=True, also returns:
            results_by_cost: {cost_bps: {model_name: backtest_df}}
    """
    results_by_cost: dict[float, dict[str, pd.DataFrame]] = {}
    rows: list[dict] = []

    for cost in cost_grid_bps:
        ls_results: dict[str, pd.DataFrame] = {}
        for model_name, pred_col in model_cols.items():
            ls_results[model_name] = backtest_long_short_forward(
                predictions=predictions,
                model_col=pred_col,
                date_col=date_col,
                id_col=id_col,
                ret_col=ret_col,
                rebalance=rebalance,
                top_frac=top_frac,
                cost_bps=float(cost),
            )

        # diagnostics table (gross + net)
        diag = generate_diagnostics_table(
            ls_results,
            rf=rf,
            periods_per_year=periods_per_year,
        )

        # keep only net rows, attach cost
        diag = diag.copy()
        diag["cost_bps"] = float(cost)
        diag = diag[diag["side"] == "net"]

        rows.append(diag)

        if return_results:
            results_by_cost[float(cost)] = ls_results

    cost_table = (
        pd.concat(rows, ignore_index=True)
        .loc[
            :,
            [
                "cost_bps",
                "model",
                "ann_return",
                "ann_vol",
                "sharpe",
                "max_drawdown",
                "avg_turnover",
                "ann_turnover",
            ],
        ]
        .sort_values(["cost_bps", "model"])
        .reset_index(drop=True)
    )

    if return_results:
        return cost_table, results_by_cost

    return cost_table
