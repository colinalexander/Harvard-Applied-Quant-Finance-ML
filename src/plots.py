"""Plotting utilities for the Harvard notebook (signals + portfolios).

This module is intentionally lightweight and notebook-facing:
- no heavy computation
- input objects must already be in 'artifact' form
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import add_size_value_ports

MODEL_COLORS = {
    "Ridge": "#4C6FFF",
    "RandomForest": "#45B29D",
    "NeuralNetwork": "#F5A623",
    "XGBoost": "#C65D7B",
}


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name}: missing columns {missing}. Available: {list(df.columns)}"
        )


def plot_ic_timeseries(
    ic_series: pd.DataFrame,
    *,
    title: str = "IC Time Series (Daily)",
) -> None:
    """Plot daily IC time series for each model.

    Expects ic_series with columns:
      - Date
      - Ridge, RandomForest, NeuralNetwork (numeric)
    """
    _require_columns(ic_series, ["Date"], "plot_ic_timeseries")

    df = ic_series.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    model_cols = [c for c in df.columns if c != "Date"]
    if not model_cols:
        raise ValueError(
            "plot_ic_timeseries: no model columns found (expected model IC columns)."
        )

    # Coerce model columns numeric
    for c in model_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    plt.figure(figsize=(10, 4))
    for model in model_cols:
        plt.plot(
            df["Date"],
            df[model],
            label=model,
            color=MODEL_COLORS.get(model, None),
            linewidth=1.2,
        )
    plt.axhline(0, color="k", linewidth=0.8, linestyle="--")
    plt.title(title)
    plt.ylabel("IC (Spearman)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_ic_timeseries(
    ic_rolling: pd.DataFrame,
    *,
    title: str = "Rolling IC (Trailing Mean)",
) -> None:
    """Plot rolling IC time series for each model (dynamic emphasis).

    Expects ic_rolling with columns:
      - Date
      - <Model>_roll (e.g., Ridge_roll)

    The strongest model (highest mean rolling IC) is highlighted with a thicker line.
    """
    if "Date" not in ic_rolling.columns:
        raise KeyError("plot_rolling_ic_timeseries: missing 'Date' column.")

    df = ic_rolling.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    roll_cols = [c for c in df.columns if c.endswith("_roll")]
    if not roll_cols:
        raise ValueError(
            "plot_rolling_ic_timeseries: no '_roll' columns found. "
            "Expected rolling IC columns like 'Ridge_roll'."
        )

    # Coerce numeric and compute means to determine the strongest model dynamically.
    roll_means = {}
    for col in roll_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        roll_means[col] = float(s.mean(skipna=True))

    best_col = max(
        roll_means, key=roll_means.get
    )  # rolling column name (e.g., Ridge_roll)
    best_model = best_col.replace("_roll", "")

    # Styling knobs
    lw_best = 2.8
    lw_other = 1.6
    alpha_best = 1.0
    alpha_other = 0.55

    plt.figure(figsize=(10, 4))

    # Plot non-best first so the highlighted line sits on top.
    for col in roll_cols:
        model = col.replace("_roll", "")
        if model == best_model:
            continue
        plt.plot(
            df["Date"],
            pd.to_numeric(df[col], errors="coerce"),
            label=model,
            color=MODEL_COLORS.get(model, None),
            linewidth=lw_other,
            alpha=alpha_other,
        )

    # Plot best last (on top)
    plt.plot(
        df["Date"],
        pd.to_numeric(df[best_col], errors="coerce"),
        label=best_model,
        color=MODEL_COLORS.get(best_model, None),
        linewidth=lw_best,
        alpha=alpha_best,
    )

    plt.axhline(0, color="k", linewidth=0.8, linestyle="--")
    plt.title(f"{title} — Highlighted: {best_model}")
    plt.ylabel("IC (Spearman)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mean_ic_bars(
    summary_table: pd.DataFrame,
    *,
    title: str = "Mean IC by Model",
) -> pd.DataFrame:
    """Plot mean IC bars and return the (sorted) summary table.

    Expects summary_table with columns:
      - model
      - mean_ic
    """
    if not {"model", "mean_ic"}.issubset(summary_table.columns):
        raise KeyError(
            "plot_mean_ic_bars: expected columns ['model', 'mean_ic'], "
            f"got {list(summary_table.columns)}"
        )

    table = summary_table.copy()
    table["mean_ic"] = pd.to_numeric(table["mean_ic"], errors="coerce")
    table = table.sort_values("mean_ic", ascending=False).reset_index(drop=True)

    labels = table["model"].tolist()
    values = table["mean_ic"].to_numpy()

    plt.figure(figsize=(6, 4))
    plt.bar(
        labels,
        values,
        color=[MODEL_COLORS.get(m, "#999999") for m in labels],
    )
    plt.axhline(0, color="k", linewidth=0.8)
    plt.ylabel("Mean IC")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return None


def plot_size_value_returns_heatmap(
    portfolio_returns: pd.DataFrame,
    *,
    annualize: bool = False,
    trading_days: int = 252,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot a Size × Value heatmap of average portfolio returns.

    The heatmap reports arithmetic mean returns for each ME × BM portfolio.
    By default, values are daily means. Annualization is optional and explicit.

    Args:
        portfolio_returns: DataFrame with columns ['Date', 'PortfolioPair', 'Return'].
        annualize: If True, multiply mean daily returns by trading_days.
        trading_days: Number of trading days used for annualization.
        figsize: Figure size passed to matplotlib.
    """
    # Build 10×10 grid (ME × BM)
    size_value_returns = (
        portfolio_returns.groupby("PortfolioPair")["Return"]
        .mean()
        .reset_index()
        .pipe(add_size_value_ports)  # must add size_port, value_port columns
        .pivot_table(
            index="value_port",
            columns="size_port",
            values="Return",
            aggfunc="mean",
        )
    )

    # Labels + scaling
    if annualize:
        size_value_returns = size_value_returns * trading_days
        cbar_label = f"Annualized Return (≈ mean × {trading_days})"
        title_suffix = " (Annualized Mean)"
        # scale the color range to match annualization
        vmin, vmax = -0.002 * trading_days, 0.002 * trading_days
        fmt = ".2f"
    else:
        cbar_label = "Average Daily Return"
        title_suffix = " (Daily Mean)"
        vmin, vmax = -0.0005, 0.0005
        fmt = ".4f"

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        size_value_returns,
        annot=True,
        fmt=fmt,
        cmap="RdYlGn",
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        mask=size_value_returns.isna(),
        cbar_kws={"label": cbar_label},
    )

    # Ensure colorbar label matches cbar_kws (and avoid duplication/inconsistency)
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label, rotation=270, labelpad=18)

    # Title dates (requires Date column)
    start_date = pd.to_datetime(portfolio_returns["Date"]).min().strftime("%Y-%m-%d")
    end_date = pd.to_datetime(portfolio_returns["Date"]).max().strftime("%Y-%m-%d")

    plt.title(
        "Average Returns by Size and Book-to-Market Portfolio"
        f"{title_suffix}\n{start_date} through {end_date}"
    )

    # Axis labels (explicit, as requested)
    plt.xlabel("Size Portfolio (Market Equity; ME01 = Small → ME10 = Large)")
    plt.ylabel("Value Portfolio (Book-to-Market; BM01 = Low → BM10 = High)")

    plt.tight_layout()
    plt.show()


def plot_size_value_marginal_means(
    portfolio_returns: pd.DataFrame,
    *,
    annualize: bool = False,
    trading_days: int = 252,
    figsize: tuple[int, int] = (12, 4),
) -> pd.DataFrame:
    """
    Plot marginal mean returns by size and value portfolio (ME and BM).

    Args:
        portfolio_returns: Long-form DataFrame with at least columns
            ['PortfolioPair', 'Return'] and optionally ['Date','Source'].
        annualize: If True, multiply mean daily returns by trading_days.
        trading_days: Trading days used for annualization.
        figsize: Figure size.

    Returns:
        A DataFrame with two Series: mean_by_size, mean_by_value (for logging/debug).
    """
    # Minimal columns only (avoid accidental work on large frames)
    df = portfolio_returns[["PortfolioPair", "Return"]].copy()

    # Mean return per portfolio pair (100 rows)
    port_mean = df.groupby("PortfolioPair", sort=False)["Return"].mean().reset_index()

    # Add size/value bucket labels (must create size_port and value_port)
    port_mean = add_size_value_ports(port_mean)

    # Create 10×10 grid of mean returns
    grid = port_mean.pivot_table(
        index="value_port", columns="size_port", values="Return", aggfunc="mean"
    ).sort_index()

    # Marginals
    mean_by_size = grid.mean(axis=0)
    mean_by_value = grid.mean(axis=1)

    if annualize:
        mean_by_size = mean_by_size * trading_days
        mean_by_value = mean_by_value * trading_days
        ylab = f"Annualized Return (≈ mean × {trading_days})"
    else:
        ylab = "Average Daily Return"

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    mean_by_size.plot(kind="bar", ax=axes[0], title="Mean Return by Size Portfolio")
    axes[0].set_xlabel("Size Portfolio\n(Market Equity; ME01 = Small → ME10 = Large)")
    axes[0].set_ylabel(ylab)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)

    mean_by_value.plot(kind="bar", ax=axes[1], title="Mean Return by Value Portfolio")
    axes[1].set_xlabel("Value Portfolio\n(Book-to-Market; BM01 = Low → BM10 = High)")
    axes[1].set_ylabel(ylab)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    plt.show()

    return None


def plot_portfolio_curves(ls_results_by_model: dict[str, pd.DataFrame]) -> None:
    """Plot cumulative gross and net curves for each model."""
    plt.figure(figsize=(10, 5))
    for model, df in ls_results_by_model.items():
        if df is None or df.empty:
            continue
        gross_curve = (1 + df["gross_ret"]).cumprod()
        net_curve = (1 + df["net_ret"]).cumprod()
        net_line = plt.plot(
            df.index,
            net_curve,
            label=f"{model} (net)",
            color=MODEL_COLORS.get(model),
        )[0]
        plt.plot(
            df.index,
            gross_curve,
            linestyle="--",
            alpha=0.5,
            label=f"{model} (gross)",
            color=net_line.get_color(),
        )
    plt.title("Cumulative Returns")
    plt.ylabel("Cumulative Growth")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_xgboost_comparison(
    artifacts,
    *,
    top_n: int = 12,
    title_prefix: str = "XGBoost Extension",
) -> None:
    """Visualize the optional boosted-tree extension (IC + feature importance)."""
    if artifacts is None:
        raise ValueError("plot_xgboost_comparison: artifacts is None.")

    def _fetch(name: str, default=None):
        if hasattr(artifacts, name):
            return getattr(artifacts, name)
        if isinstance(artifacts, Mapping) and name in artifacts:
            return artifacts[name]
        return default

    predictions = _fetch("predictions")
    feature_importance = _fetch("feature_importance")
    ic_by_date = _fetch("ic_by_date")
    ic_series = _fetch("ic_series")
    ic_rolling = _fetch("ic_rolling")
    summary_table = _fetch("summary_table")
    backend = _fetch("backend", "xgboost")

    if predictions is None:
        raise ValueError("plot_xgboost_comparison: predictions missing from artifacts.")
    _require_columns(predictions, ["Actual", "pred_xgb"], "plot_xgboost_comparison")

    color = MODEL_COLORS.get("XGBoost", "#C65D7B")

    # Prefer ic_by_date; fallback to ic_series; otherwise compute from predictions.
    if ic_by_date is None and ic_series is not None:
        ic_by_date = ic_series.rename(columns={"ic": "IC", "ic_roll": "IC_rolling"})
    if ic_by_date is None and "Date" in predictions.columns:
        records = []
        for d, g in predictions.groupby("Date", sort=True):
            ic_val = pd.Series(g["Actual"]).corr(
                pd.Series(g["pred_xgb"]), method="spearman"
            )
            records.append({"Date": pd.to_datetime(d), "IC": ic_val})
        ic_by_date = pd.DataFrame(records).sort_values("Date")

    if ic_rolling is None and ic_by_date is not None and not ic_by_date.empty:
        ic_rolling = ic_by_date[["Date", "IC"]].copy()
        ic_rolling["XGBoost"] = (
            ic_rolling["IC"].rolling(window=126, min_periods=126).mean()
        )

    if ic_by_date is not None and not ic_by_date.empty:
        df = ic_by_date.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["IC"] = pd.to_numeric(df["IC"], errors="coerce")
        plt.figure(figsize=(10, 3.8))
        plt.plot(df["Date"], df["IC"], color=color, linewidth=1.3, label="Daily IC")
        if ic_rolling is not None and not getattr(ic_rolling, "empty", True):
            roll = ic_rolling.copy()
            roll["Date"] = pd.to_datetime(roll["Date"])
            roll["XGBoost"] = pd.to_numeric(roll.iloc[:, -1], errors="coerce")
            plt.plot(
                roll["Date"],
                roll.iloc[:, -1],
                color=color,
                linewidth=2.2,
                alpha=0.7,
                label="Rolling IC",
            )
        plt.axhline(0, color="k", linewidth=0.8, linestyle="--")
        plt.title(f"{title_prefix}: Daily IC")
        plt.ylabel("IC (Spearman)")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # IC histogram
    if ic_by_date is not None and not ic_by_date.empty:
        plt.figure(figsize=(6, 4))
        plt.hist(ic_by_date["IC"].dropna().to_numpy(), bins=60, color=color, alpha=0.85)
        plt.title("XGBoost — IC Distribution")
        plt.xlabel("IC (Spearman)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # Feature importance (optional)
    if feature_importance is not None and not getattr(
        feature_importance, "empty", True
    ):
        top = feature_importance.head(top_n).copy()
        top = top.iloc[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(
            top["feature"].astype(str),
            pd.to_numeric(top["importance"], errors="coerce"),
            color=color,
            alpha=0.85,
        )
        plt.title(f"Top {len(top)} Features ({backend})")
        plt.xlabel("Average Importance")
        plt.tight_layout()
        plt.show()

    if summary_table is not None and not getattr(summary_table, "empty", True):
        plt.figure(figsize=(4, 3.2))
        plt.bar(summary_table["model"], summary_table["mean_ic"], color=color)
        plt.axhline(0, color="k", linewidth=0.8)
        plt.title("Mean IC (XGBoost)")
        plt.ylabel("Mean IC")
        plt.tight_layout()
        plt.show()
