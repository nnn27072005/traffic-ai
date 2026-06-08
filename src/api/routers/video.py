# src/api/routers/video.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query, Form
from sqlalchemy.orm import Session
import os
import uuid
from src.api import crud, schemas, database, pipeline
from src.api.video_processor import process_video_file

router = APIRouter(prefix="/video", tags=["Video Analysis"])

# Temporary directory for uploaded videos
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=schemas.SessionResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    save_output_form: bool | None = Form(None, alias="save_output"),
    save_evidence_form: bool | None = Form(None, alias="save_evidence"),
    analysis_fps_form: float | None = Form(None, alias="analysis_fps"),
    save_output_query: bool | None = Query(None, alias="save_output"),
    save_evidence_query: bool | None = Query(None, alias="save_evidence"),
    analysis_fps_query: float | None = Query(None, alias="analysis_fps"),
    db: Session = Depends(database.get_db)
):
    """
    Upload a video file and start a background processing session.
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    save_output = save_output_form if save_output_form is not None else bool(save_output_query)
    save_evidence = save_evidence_form if save_evidence_form is not None else bool(save_evidence_query)
    analysis_fps = analysis_fps_form if analysis_fps_form is not None else analysis_fps_query
    
    # Save file locally
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")
    
    # Create session
    session = crud.create_session(
        db,
        source_type="upload",
        source_path=file_path,
        config={
            "filename": file.filename,
            "save_output": save_output,
            "save_evidence": save_evidence,
            "analysis_fps": analysis_fps,
        }
    )
    
    # Start background processing
    from src.api.main import pipeline as global_pipeline
    if global_pipeline is None:
        raise HTTPException(status_code=503, detail="Neural Engine is still loading")
        
    background_tasks.add_task(
        process_video_file,
        video_path=file_path,
        session_id=session.id,
        pipeline=global_pipeline,
        target_analysis_fps=analysis_fps,
        enable_export=save_output,
        save_evidence=save_evidence,
    )
    
    return session

@router.post("/url", response_model=schemas.SessionResponse)
async def process_video_url(
    request: schemas.VideoUrlRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db)
):
    """
    Start a background processing session from a video URL (YouTube or direct link).
    """
    from src.api.utils.video import get_stream_url
    
    try:
        stream_url = get_stream_url(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Create session
    session = crud.create_session(
        db,
        source_type="stream" if request.stream_mode else "url",
        source_path=stream_url,
        config={
            "original_url": request.url,
            "stream_mode": request.stream_mode,
            "save_output": request.save_output,
            "save_evidence": request.save_evidence,
            "analysis_fps": request.analysis_fps,
        }
    )
    
    # Start background processing
    from src.api.main import pipeline as global_pipeline
    if global_pipeline is None:
        raise HTTPException(status_code=503, detail="Neural Engine is still loading")
        
    background_tasks.add_task(
        process_video_file,
        video_path=stream_url,
        session_id=session.id,
        pipeline=global_pipeline,
        target_analysis_fps=request.analysis_fps,
        enable_export=request.save_output,
        save_evidence=request.save_evidence,
    )
    
    return session
