from __future__ import annotations

import pandas as pd


def add_size_value_ports(df: pd.DataFrame, col: str = "PortfolioPair") -> pd.DataFrame:
    """Add size/value labels from a PortfolioPair column (e.g., 'ME01 BM10')."""
    df[["size_port", "value_port"]] = df[col].str.split(" ", expand=True)
    return df
