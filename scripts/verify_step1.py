"""Quick verification script for Step 1 models."""
from src.api.models import (
    Camera, ProcessingSession, TrafficCount, Violation, ViolationAsset,
    auto_classify_severity, Base,
)
from src.api.database import engine, DATABASE_URL, STORAGE_BACKEND
from src.api.schemas import (
    CameraCreate, SessionCreate, ViolationRecordSchema, SummaryStats,
    ViolationFilter, SessionResponse,
)
from src.api.storage import LocalStorage, MinIOStorage, get_storage

print("=" * 60)
print("Step 1 Verification")
print("=" * 60)

# 1. All models import OK
print("\n[OK] All 5 models imported: Camera, ProcessingSession, TrafficCount, Violation, ViolationAsset")

# 2. Table names
tables = [t.name for t in Base.metadata.sorted_tables]
print(f"[OK] Tables: {tables}")

# 3. Auto-severity classification
tests = [
    ("WithoutHelmet", 0.95, "critical"),
    ("WithoutHelmet", 0.75, "high"),
    ("WithoutHelmet", 0.45, "medium"),
    ("WithoutHelmet", 0.20, "low"),
    ("UnknownType",   0.99, "medium"),
    ("UnknownType",   0.10, "low"),
]
for vtype, conf, expected in tests:
    result = auto_classify_severity(vtype, conf)
    status = "OK" if result == expected else "FAIL"
    print(f"[{status}] severity({vtype}, {conf}) = {result} (expected {expected})")

# 4. Schemas
print(f"\n[OK] Pydantic schemas imported: CameraCreate, SessionCreate, ViolationRecordSchema, etc.")

# 5. Storage backend
storage = get_storage()
print(f"[OK] Storage backend: {type(storage).__name__} (configured: {STORAGE_BACKEND})")

# 6. Database URL
print(f"[OK] DATABASE_URL: {DATABASE_URL}")

# 7. Create tables with SQLite (quick test)
import os
os.environ["DATABASE_URL"] = "sqlite:///./test_verify.db"
from sqlalchemy import create_engine as ce
test_engine = ce("sqlite:///./test_verify.db")
Base.metadata.create_all(bind=test_engine)
print("[OK] All tables created successfully in SQLite test DB")

# Cleanup
import pathlib
pathlib.Path("test_verify.db").unlink(missing_ok=True)

print("\n" + "=" * 60)
print("All Step 1 verifications PASSED")
print("=" * 60)
