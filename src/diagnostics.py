import pandas as pd


def compute_portfolio_return_stats(
    returns: pd.DataFrame,
    *,
    date_col: str = "Date",
    portfolio_col: str = "PortfolioPair",
    return_col: str = "Return",
) -> pd.DataFrame:
    """Compute per-portfolio distribution stats for daily excess returns.

    Output columns:
      PortfolioPair, n_obs, mean, std, min, max, skew, excess_kurtosis
    """
    required = {date_col, portfolio_col, return_col}
    missing = required - set(returns.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = returns[[date_col, portfolio_col, return_col]].copy()
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df = df.dropna(subset=[return_col])

    g = df.groupby(portfolio_col)[return_col]

    df = (
        pd.DataFrame(
            {
                "n_obs": g.count(),
                "mean": g.mean(),
                "std": g.std(),
                "min": g.min(),
                "max": g.max(),
                "skew": g.apply(lambda s: s.skew()),
                # pandas Series.kurt() is *excess kurtosis* by default
                "excess_kurtosis": g.apply(lambda s: s.kurt()),
            }
        )
        .reset_index()
        .sort_values(portfolio_col)
        .reset_index(drop=True)
    )
    return df


def compute_portfolio_return_stats_summary(
    portfolio_return_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Compute cross-portfolio summary of portfolio return stats."""
    stat_cols = ["n_obs", "mean", "std", "min", "max", "skew", "excess_kurtosis"]

    missing = set(stat_cols) - set(portfolio_return_stats)
    if missing:
        raise KeyError(f"`portfolio_return_stats` missing columns: {missing}")

    df = (
        portfolio_return_stats[stat_cols]
        .agg(["min", "max", "mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "stat"})
    )
    return df
