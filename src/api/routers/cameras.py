# src/api/routers/cameras.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.api import crud, schemas, database

router = APIRouter(prefix="/cameras", tags=["Cameras"])

@router.get("/", response_model=list[schemas.CameraResponse])
def read_cameras(active_only: bool = False, db: Session = Depends(database.get_db)):
    return crud.get_cameras(db, active_only=active_only)

@router.post("/", response_model=schemas.CameraResponse)
def create_camera(camera: schemas.CameraCreate, db: Session = Depends(database.get_db)):
    return crud.create_camera(db, name=camera.name, **camera.dict())

@router.get("/{camera_id}", response_model=schemas.CameraResponse)
def read_camera(camera_id: int, db: Session = Depends(database.get_db)):
    db_camera = crud.get_camera(db, camera_id=camera_id)
    if db_camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return db_camera
