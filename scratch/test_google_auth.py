
import sys
import os
from unittest.mock import MagicMock, patch

# Mocking the environment before importing
os.environ["GOOGLE_CLIENT_ID"] = "test-client-id"
os.environ["DATABASE_URL"] = "sqlite:///./test_verify.db"

# Add the project root to sys.path
sys.path.append(os.path.abspath("."))

from src.api import models, database, schemas
from src.api.routers.auth import google_auth
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup test database
engine = create_engine("sqlite:///./test_verify.db")
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
models.Base.metadata.create_all(bind=engine)

def test_google_auth_new_user():
    db = TestingSessionLocal()
    # Clean up
    db.query(models.User).delete()
    db.commit()

    mock_idinfo = {
        "email": "testuser@gmail.com",
        "name": "Test User",
        "sub": "123456789",
        "picture": "http://example.com/pic.jpg"
    }

    request = schemas.GoogleLoginRequest(token="mock-token")

    with patch("google.oauth2.id_token.verify_oauth2_token", return_value=mock_idinfo):
        response = google_auth(request, db)
        
        assert response["user"].email == "testuser@gmail.com"
        assert response["user"].google_id == "123456789"
        assert "access_token" in response
        print("Success: New user created via Google Auth")

    db.close()

def test_google_auth_existing_user_link():
    db = TestingSessionLocal()
    # Create user with same email but no google_id
    db.query(models.User).delete()
    user = models.User(username="testuser", email="testuser@gmail.com", hashed_password="...")
    db.add(user)
    db.commit()

    mock_idinfo = {
        "email": "testuser@gmail.com",
        "name": "Test User",
        "sub": "123456789",
        "picture": "http://example.com/pic.jpg"
    }

    request = schemas.GoogleLoginRequest(token="mock-token")

    with patch("google.oauth2.id_token.verify_oauth2_token", return_value=mock_idinfo):
        response = google_auth(request, db)
        
        assert response["user"].email == "testuser@gmail.com"
        assert response["user"].google_id == "123456789"
        assert response["user"].username == "testuser"
        print("Success: Existing user linked with Google ID")

    db.close()

if __name__ == "__main__":
    try:
        test_google_auth_new_user()
        test_google_auth_existing_user_link()
        print("All backend auth tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
