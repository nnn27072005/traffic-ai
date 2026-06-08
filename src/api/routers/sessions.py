# src/api/routers/sessions.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.api import crud, schemas, database

router = APIRouter(prefix="/sessions", tags=["Sessions"])

@router.post("/", response_model=schemas.SessionResponse)
def create_session(session_in: schemas.SessionCreate, db: Session = Depends(database.get_db)):
    return crud.create_session(
        db,
        camera_id=session_in.camera_id,
        source_type=session_in.source_type,
        source_path=session_in.source_path,
        config=session_in.config
    )

@router.get("/{session_id}", response_model=schemas.SessionResponse)
def read_session(session_id: int, db: Session = Depends(database.get_db)):
    db_session = crud.get_session(db, session_id=session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return db_session

@router.get("/", response_model=list[schemas.SessionResponse])
def read_sessions(skip: int = 0, limit: int = 20, db: Session = Depends(database.get_db)):
    return crud.get_sessions(db, skip=skip, limit=limit)

@router.post("/{session_id}/stop", response_model=schemas.SessionResponse)
def stop_session(session_id: int, db: Session = Depends(database.get_db)):
    db_session = crud.stop_session(db, session_id=session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return db_session
