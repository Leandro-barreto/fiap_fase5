"""
utils.py – helper functions for data preparation and transformation
------------------------------------------------------------------

These functions convert raw input data into a format suitable for
consumption by the machine learning model and back again.  They are
largely adapted from the original repository's implementation【200823175972727†L6-L31】.

You may need to adjust these functions depending on the format of
your attached file or the specifics of your trained model.  For
example, if your model uses a different feature set or scaling
approach, update ``prepare_input`` and ``inverse_transform_output``
accordingly.  The ``fetch_data`` function relies on yfinance to
retrieve recent history when no explicit data is provided.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List, Tuple

from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


def prepare_input(data: List[dict], lookback: int = 60) -> Tuple[np.ndarray, MinMaxScaler]:
    """Prepare input data for model prediction.

    The input list of dictionaries should have keys corresponding to
    feature names (e.g., ``Open``, ``High``, etc.).  The data is
    transformed into a 3D array of shape ``(1, lookback, n_features)``
    suitable for LSTM models and a fitted ``MinMaxScaler`` is returned
    for inverse scaling of predictions.
    """
    df = pd.DataFrame(data)
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X_input = np.expand_dims(scaled[-lookback:], axis=0)

    return X_input, scaler


def inverse_transform_output(predicted_scaled: np.ndarray, scaler: MinMaxScaler, close_index: int = 3, total_features: int = 5) -> np.ndarray:
    """Convert scaled prediction back to the original price scale.

    Only the closing price is transformed back; other feature columns
    are set to zero before applying the inverse transform.  This
    function mirrors the behaviour of
    :func:`inverse_transform_close` from the original code【200823175972727†L17-L20】.
    """
    zeros = np.zeros((len(predicted_scaled), total_features))
    zeros[:, close_index] = predicted_scaled.flatten()
    return scaler.inverse_transform(zeros)[:, close_index]


def fetch_data(ticker: str, lookback: int = 60, end_date: date | None = None) -> Tuple[pd.DataFrame, date]:
    """Fetch recent historical data for a given ticker using yfinance.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol to download data for.
    lookback : int, optional
        Number of days of historical data to return, by default 60.
    end_date : date, optional
        The end date for the data.  If ``None``, today's date is used.

    Returns
    -------
    tuple
        A tuple containing a DataFrame of the last ``lookback`` rows and
        the end date used in the query.
    """
    if end_date is None:
        end_date = date.today()

    start_date = end_date - timedelta(days=lookback + 10)
    df = yf.download(ticker, end=str(end_date), start=str(start_date))

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns.name = None

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df[-lookback:], end_date
