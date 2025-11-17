"""
Data I/O utilities for S&P 500 project.

This module:
- loads the three core tables (stocks, companies, index)
- aligns dates and performs basic cleaning
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from . import config


def load_prices(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load per-ticker stock price data.

    Expected columns in sp500_stocks.csv (you can adapt after inspecting the file):
        - date / Date
        - ticker or symbol (we normalize to 'ticker')
        - adj_close / Adj Close / close (we normalize to 'adj_close')

    Returns
    -------
    DataFrame with columns: ['date', 'ticker', 'adj_close', ...]
    """
    if path is None:
        path = config.SP500_STOCKS_FILE

    df = pd.read_csv(path)
    # Keep original names but also create a lower-case copy for matching
    cols_lower = {c.lower(): c for c in df.columns}

    # --- date column ---
    if "date" in cols_lower:
        date_col = cols_lower["date"]
    else:
        raise ValueError(
            f"Expected a 'date' column in stock data, got columns: {list(df.columns)}"
        )
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})

    # --- ticker / symbol column ---
    ticker_col = None
    for candidate in ["ticker", "symbol"]:
        if candidate in cols_lower:
            ticker_col = cols_lower[candidate]
            break
    if ticker_col is None:
        raise ValueError(
            "Expected a 'ticker' or 'symbol' column in stock data, "
            f"got columns: {list(df.columns)}"
        )
    df = df.rename(columns={ticker_col: "ticker"})

    # --- adjusted close / close column ---
    adj_candidates = ["adj close", "adj_close", "adjclose", "close"]
    adj_col = None
    for candidate in adj_candidates:
        if candidate in cols_lower:
            adj_col = cols_lower[candidate]
            break
    if adj_col is None:
        raise ValueError(
            "Could not find an adjusted price column. "
            "Expected one of: 'Adj Close', 'adj_close', 'adjclose', 'Close'. "
            f"Got columns: {list(df.columns)}"
        )
    df = df.rename(columns={adj_col: "adj_close"})

    # Keep only relevant columns plus any others you might need
    # (here we keep everything; downstream code only uses date/ticker/adj_close)
    return df


def load_companies(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load company metadata.

    Expected columns (you can adapt as needed):
        - ticker or symbol (we normalize to index 'ticker')
        - sector
        - industry / gics
        - etc.

    Returns
    -------
    DataFrame indexed by ticker.
    """
    if path is None:
        path = config.SP500_COMPANIES_FILE

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    ticker_col = None
    for candidate in ["ticker", "symbol"]:
        if candidate in cols_lower:
            ticker_col = cols_lower[candidate]
            break
    if ticker_col is None:
        raise ValueError(
            "Expected a 'ticker' or 'symbol' column in companies data, "
            f"got columns: {list(df.columns)}"
        )

    df = df.rename(columns={ticker_col: "ticker"})
    df = df.set_index("ticker").sort_index()
    return df


def load_index_series(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load S&P 500 index level.

    Expected columns:
        - date / Date
        - index_level OR one of:
          'adj_close', 'close', 's&p500', 'sp500', '^gspc', 'price'

    Returns
    -------
    DataFrame with columns ['index_level'] and DatetimeIndex.
    """
    if path is None:
        path = config.SP500_INDEX_FILE

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    # --- date column ---
    if "date" in cols_lower:
        date_col = cols_lower["date"]
    else:
        raise ValueError(
            f"Expected a 'date' column in index data, got columns: {list(df.columns)}"
        )
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})

    # --- index level column ---
    if "index_level" in cols_lower:
        idx_col = cols_lower["index_level"]
    else:
        alt_candidates = [
            "adj_close",
            "close",
            "s&p500",
            "sp500",
            "^gspc",
            "price",
        ]
        idx_col = None
        for candidate in alt_candidates:
            if candidate in cols_lower:
                idx_col = cols_lower[candidate]
                break

        if idx_col is None:
            raise ValueError(
                "Could not infer index level column in index data. "
                "Expected one of: 'index_level', 'Adj Close', 'Close', "
                "'S&P500', 'sp500', '^GSPC', 'price'. "
                f"Got columns: {list(df.columns)}"
            )

    df = df.rename(columns={idx_col: "index_level"})
    df = df.sort_values("date").set_index("date")
    return df[["index_level"]]


def build_price_panel(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot tidy prices into a wide panel:
        index = date, columns = ticker, values = adj_close.

    Parameters
    ----------
    prices_df : DataFrame
        Output of load_prices().

    Returns
    -------
    price_panel : DataFrame
        index = date, columns = tickers, values = adjusted close.
    """
    required = {"date", "ticker", "adj_close"}
    missing = required.difference(prices_df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in prices_df: {missing}")

    panel = (
        prices_df
        .pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )
    return panel
