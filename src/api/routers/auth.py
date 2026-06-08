# src/api/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from src.api import crud, database, models, schemas, auth
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os

router = APIRouter(tags=["Authentication"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

@router.post("/auth/register", response_model=schemas.UserResponse)
def register_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = auth.get_password_hash(user.password)
    new_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.post("/auth/login", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "user": user}

@router.post("/auth/google", response_model=schemas.Token)
def google_auth(request: schemas.GoogleLoginRequest, db: Session = Depends(database.get_db)):
    try:
        # Verify the token with Google
        idinfo = id_token.verify_oauth2_token(
            request.token, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10
        )
        
        email = idinfo.get("email")
        name = idinfo.get("name")
        google_id = idinfo.get("sub")
        picture = idinfo.get("picture")
        
        if not email:
            raise HTTPException(status_code=400, detail="Google token missing email")

        user = db.query(models.User).filter(models.User.google_id == google_id).first()
        if not user:
            # Check if a user with this email already exists but doesn't have a google_id linked
            user = db.query(models.User).filter(models.User.email == email).first()
            if user:
                user.google_id = google_id
                user.avatar_url = picture
            else:
                # Create new user
                user = models.User(
                    username=name.replace(" ", "_").lower(),
                    full_name=name,
                    email=email,
                    google_id=google_id,
                    avatar_url=picture
                )
                db.add(user)
            
            db.commit()
            db.refresh(user)
        
        access_token = auth.create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer", "user": user}
        
    except ValueError as e:
        print(f"Google Token Verification Failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid Google Token: {str(e)}")

@router.get("/auth/me", response_model=schemas.UserResponse)
def get_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user
