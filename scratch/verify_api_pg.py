import sys
from sqlalchemy import text

sys.path.insert(0, ".")
from src.api.database import SessionLocal, engine

def verify():
    print("Testing connection through database.py config...")
    db = SessionLocal()
    try:
        # Check users count
        res = db.execute(text("SELECT COUNT(*) FROM users")).scalar()
        print(f"[OK] Users count in Cloud DB: {res}")
        
        # Check violations count
        res = db.execute(text("SELECT COUNT(*) FROM violations")).scalar()
        print(f"[OK] Violations count in Cloud DB: {res}")
        
        # Check processing sessions count
        res = db.execute(text("SELECT COUNT(*) FROM processing_sessions")).scalar()
        print(f"[OK] Processing sessions count in Cloud DB: {res}")
        
        # Retrieve some violations
        res = db.execute(text("SELECT id, track_id, violation_type, confidence FROM violations LIMIT 5")).fetchall()
        print("Sample violations:")
        for r in res:
            print(f"  ID: {r[0]}, Track: {r[1]}, Type: {r[2]}, Confidence: {r[3]:.2f}")
            
    except Exception as e:
        print("Error:", e)
    finally:
        db.close()

if __name__ == "__main__":
    verify()
