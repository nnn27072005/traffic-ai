# src/api/routers/violations.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from src.api import crud, schemas, database
from typing import Optional
from datetime import datetime

router = APIRouter(prefix="/violations", tags=["Violations"])

@router.get("/", response_model=list[schemas.ViolationRecordSchema])
def read_violations(
    skip: int = 0,
    limit: int = 100,
    camera_id: Optional[int] = None,
    session_id: Optional[int] = None,
    violation_type: Optional[str] = None,
    severity: Optional[str] = None,
    reviewed: Optional[bool] = None,
    db: Session = Depends(database.get_db)
):
    violations = crud.get_violations(
        db, 
        skip=skip, 
        limit=limit,
        camera_id=camera_id,
        session_id=session_id,
        violation_type=violation_type,
        severity=severity,
        reviewed=reviewed
    )
    return violations

@router.get("/{violation_id}", response_model=schemas.ViolationRecordSchema)
def read_violation(violation_id: int, db: Session = Depends(database.get_db)):
    violation = crud.get_violation(db, violation_id=violation_id)
    if violation is None:
        raise HTTPException(status_code=404, detail="Violation not found")
    return violation

@router.patch("/{violation_id}/review", response_model=schemas.ViolationRecordSchema)
def review_violation(
    violation_id: int, 
    review: schemas.ViolationReview, 
    db: Session = Depends(database.get_db)
):
    violation = crud.review_violation(
        db, 
        violation_id=violation_id,
        reviewed=review.reviewed,
        severity=review.severity,
        notes=review.notes
    )
    if violation is None:
        raise HTTPException(status_code=404, detail="Violation not found")
    return violation
