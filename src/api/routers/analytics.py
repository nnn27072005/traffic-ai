# src/api/routers/analytics.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.api import crud, schemas, database

router = APIRouter(prefix="/stats", tags=["Analytics"])

@router.get("/summary", response_model=schemas.SummaryStats)
def get_summary(db: Session = Depends(database.get_db)):
    return crud.get_summary_stats(db)

@router.get("/history", response_model=list[schemas.TrafficTimeSeriesPoint])
def get_history(limit: int = 100, db: Session = Depends(database.get_db)):
    # Reusing get_traffic_time_series from crud
    points = crud.get_traffic_time_series(db, limit=limit)
    return points
