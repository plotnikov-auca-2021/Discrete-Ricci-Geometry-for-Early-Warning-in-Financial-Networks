"""
Data I/O utilities for S&P 500 project.

This module:
- loads the three core tables (stocks, companies, index)
- aligns dates and basic cleaning
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from . import config


def load_prices(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load per-ticker price data.

    Expected columns in sp500_stocks.csv (you can adapt after inspecting the file):
        - date
        - ticker
        - adj_close  (or 'Adj Close', etc.)

    Returns
    -------
    DataFrame with columns: ['date', 'ticker', 'adj_close', ...]
    """
    if path is None:
        path = config.SP500_STOCKS_FILE
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in stock data.")
    df["date"] = pd.to_datetime(df["date"])
    if "adj_close" not in df.columns and "adjclose" in df.columns:
        df = df.rename(columns={"adjclose": "adj_close"})
    if "adj_close" not in df.columns:
        # fall back to 'close'
        if "close" in df.columns:
            df = df.rename(columns={"close": "adj_close"})
        else:
            raise ValueError("Could not find 'adj_close' or 'close' column in stock data.")
    return df


def load_companies(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load company metadata.

    Expected columns (you can adapt as needed):
        - ticker
        - name
        - sector
        - industry / gics

    Returns
    -------
    DataFrame indexed by ticker.
    """
    if path is None:
        path = config.SP500_COMPANIES_FILE
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("Expected a 'ticker' column in companies data.")
    df = df.set_index("ticker").sort_index()
    return df


def load_index_series(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load S&P 500 index level.

    Expected columns:
        - date
        - index_level (or 'close', 'adj_close')

    Returns
    -------
    DataFrame with columns ['date', 'index_level'] and datetime index.
    """
    if path is None:
        path = config.SP500_INDEX_FILE
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in index data.")
    df["date"] = pd.to_datetime(df["date"])
    if "index_level" not in df.columns:
        # try common alternatives
        for alt in ["adj_close", "close", "sp500"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "index_level"})
                break
    if "index_level" not in df.columns:
        raise ValueError("Could not infer 'index_level' column in index data.")
    df = df.sort_values("date").set_index("date")
    return df[["index_level"]]


def build_price_panel(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot tidy prices into a wide panel: index=date, columns=ticker, values=adj_close.

    Parameters
    ----------
    prices_df : DataFrame
        Output of load_prices.

    Returns
    -------
    price_panel : DataFrame
        index = date, columns = tickers, values = adjusted close.
    """
    required = {"date", "ticker", "adj_close"}
    if not required.issubset(set(prices_df.columns)):
        raise ValueError(f"Expected at least columns {required}, got {prices_df.columns}")
    panel = prices_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return panel
