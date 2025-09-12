"""
predict.py – API endpoints for generating predictions
----------------------------------------------------

This module defines HTTP endpoints that accept input parameters
representing stock tickers, dates or historical price data and return
predictions from a pre‑trained ML model.  The general flow of a
prediction request is as follows:

1. Load the appropriate model via :func:`new_api.model.loader.load_model`.
2. Prepare the input data using :func:`new_api.utils.prepare_input`.
3. Execute the model's prediction.
4. Convert the model output back to the original scale using
   :func:`new_api.utils.inverse_transform_output`.

The code here follows the patterns established in the repository's
``api/routes/predict.py``【687098051368206†L9-L61】 but has been simplified
and documented for clarity.  You can extend these functions to
implement additional validation, error handling, or asynchronous
execution as needed.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File

from ..schemas import HistoricalData, PredictionRequest, PredictionResponse
from ..model.loader import load_model
from ..utils import prepare_input, inverse_transform_output, fetch_data


router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict the closing price for a given ticker.

    Parameters
    ----------
    request : PredictionRequest
        Includes the ticker symbol, an optional target_date and an
        optional list of historical records.  If ``data`` is supplied
        the API uses it directly; otherwise it fetches the last
        ``lookback`` records from a data source.

    Returns
    -------
    PredictionResponse
        Contains the ticker, the predicted closing price and the
        prediction date.
    """
    # 1) Load the appropriate model for the ticker.
    try:
        model = load_model(request.ticker)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modelo para '{request.ticker}' não encontrado.")

    # 2) Determine the data to use for prediction.
    if request.data:
        # Convert list of HistoricalData into the format expected by the model
        input_data, scaler = prepare_input([d.dict() if hasattr(d, "dict") else d for d in request.data])
        prediction_date: Optional[date] = request.target_date or date.today()
    else:
        # When no explicit data is provided, fetch the most recent history up to the target date.
        # Ensure the target date does not exceed today to avoid errors【687098051368206†L18-L20】.
        end_date = request.target_date or date.today()
        if end_date > date.today():
            end_date = date.today()
        df, prediction_date = fetch_data(request.ticker, end_date=end_date)
        input_data, scaler = prepare_input(df.to_dict(orient="records"))

    # 3) Make a prediction using the loaded model.
    try:
        y_pred = model.predict(input_data)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Erro ao realizar a predição: {exc}")

    # 4) Convert the model's scaled output back to original scale
    y_real = inverse_transform_output(y_pred, scaler)

    return PredictionResponse(
        ticker=request.ticker,
        predicted_close=float(y_real[0]),
        prediction_date=prediction_date,
    )


@router.post("/predict/from_csv", response_model=PredictionResponse)
async def predict_from_csv(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict the closing price using historical data contained in a CSV file.

    The CSV is expected to have columns ``Open, High, Low, Close, Volume`` and
    optionally ``Date``.  The ticker symbol is inferred from the file name
    (e.g., ``AAPL_prices.csv`` ⇒ ticker ``AAPL``).  This endpoint mirrors the
    behaviour of the original implementation【687098051368206†L34-L62】.
    """
    import pandas as pd  # Imported here to avoid unused dependency warnings

    # Try to read the uploaded file into a DataFrame
    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Erro ao ler o arquivo CSV.")

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in expected_cols):
        raise HTTPException(
            status_code=400,
            detail=f"CSV deve conter as colunas: {expected_cols}",
        )

    # Infer the ticker from the file name (before the first underscore)
    ticker = file.filename.split("_")[0].split(".")[0].upper()

    try:
        model = load_model(ticker)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modelo para '{ticker}' não encontrado.")

    input_data, scaler = prepare_input(df[expected_cols].to_dict(orient="records"))
    y_pred = model.predict(input_data)
    y_real = inverse_transform_output(y_pred, scaler)

    last_date = pd.to_datetime(df.get("Date", pd.Timestamp.today())).max().date()

    return PredictionResponse(
        ticker=ticker,
        predicted_close=float(y_real[0]),
        prediction_date=last_date,
    )
