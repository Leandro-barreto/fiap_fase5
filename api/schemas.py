"""
schemas.py – Pydantic models for request/response bodies
-------------------------------------------------------

Using Pydantic models ensures type safety and automatic documentation
for the API.  The models defined here mirror those in the
repository's ``api/schemas.py``【257970202163222†L11-L19】 but you can
extend them with additional fields as needed.
"""

from datetime import date
from typing import List, Optional

from pydantic import BaseModel


class HistoricalData(BaseModel):
    """Representation of a single historical record used for predictions."""

    Open: float
    High: float
    Low: float
    Close: float
    Volume: float


class PredictionRequest(BaseModel):
    """Request schema for the JSON prediction endpoint."""

    ticker: str
    target_date: Optional[date] = None
    data: Optional[List[HistoricalData]] = None


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""

    ticker: str
    predicted_close: float
    prediction_date: date
