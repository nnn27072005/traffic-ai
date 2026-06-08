import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import models
# We need to add the parent directory to Python path if running from root
sys.path.insert(0, ".")
from src.api.database import Base
from src.api.models import User, Camera, ProcessingSession, ViolationAsset, TrafficCount, Violation

sqlite_url = "sqlite:///./traffic_system.db"
pg_url = "postgresql://traffic_ai_user:26udsfygkrrX7ecUiZN9JrI1ET90SVri@dpg-d7p26ecm0tmc73dgfdig-a.singapore-postgres.render.com/traffic_ai"

# Create engines and session factories
sqlite_engine = create_engine(sqlite_url)
sqlite_session_factory = sessionmaker(bind=sqlite_engine)

pg_engine = create_engine(pg_url)
pg_session_factory = sessionmaker(bind=pg_engine)

def migrate():
    sqlite_session = sqlite_session_factory()
    pg_session = pg_session_factory()
    
    try:
        print("Starting migration...")
        
        # 1. Migrate Users
        print("Migrating Users...")
        sqlite_users = sqlite_session.query(User).all()
        print(f"Found {len(sqlite_users)} users in SQLite.")
        for user in sqlite_users:
            # Check if user already exists in PG
            existing = pg_session.query(User).filter_by(id=user.id).first()
            if not existing:
                # Create a new instance to avoid session bound issues
                new_user = User(
                    id=user.id,
                    username=user.username,
                    full_name=user.full_name,
                    email=user.email,
                    hashed_password=user.hashed_password,
                    google_id=user.google_id,
                    avatar_url=user.avatar_url,
                    role=user.role,
                    is_active=user.is_active,
                    created_at=user.created_at,
                    updated_at=user.updated_at
                )
                pg_session.add(new_user)
        pg_session.commit()
        print("Users migrated successfully.")

        # 2. Migrate Cameras
        print("Migrating Cameras...")
        sqlite_cameras = sqlite_session.query(Camera).all()
        print(f"Found {len(sqlite_cameras)} cameras in SQLite.")
        for camera in sqlite_cameras:
            existing = pg_session.query(Camera).filter_by(id=camera.id).first()
            if not existing:
                new_camera = Camera(
                    id=camera.id,
                    name=camera.name,
                    location=camera.location,
                    latitude=camera.latitude,
                    longitude=camera.longitude,
                    stream_url=camera.stream_url,
                    description=camera.description,
                    is_active=camera.is_active,
                    created_at=camera.created_at,
                    updated_at=camera.updated_at
                )
                pg_session.add(new_camera)
        pg_session.commit()
        print("Cameras migrated successfully.")

        # 3. Migrate ProcessingSessions
        print("Migrating ProcessingSessions...")
        sqlite_sessions = sqlite_session.query(ProcessingSession).all()
        print(f"Found {len(sqlite_sessions)} processing sessions in SQLite.")
        for sess in sqlite_sessions:
            existing = pg_session.query(ProcessingSession).filter_by(id=sess.id).first()
            if not existing:
                new_sess = ProcessingSession(
                    id=sess.id,
                    camera_id=sess.camera_id,
                    source_type=sess.source_type,
                    source_path=sess.source_path,
                    status=sess.status,
                    total_frames=sess.total_frames,
                    processed_frames=sess.processed_frames,
                    fps=sess.fps,
                    error_message=sess.error_message,
                    config=sess.config,
                    started_at=sess.started_at,
                    completed_at=sess.completed_at,
                    output_video_path=sess.output_video_path
                )
                pg_session.add(new_sess)
        pg_session.commit()
        print("ProcessingSessions migrated successfully.")

        # 4. Migrate ViolationAssets
        print("Migrating ViolationAssets...")
        sqlite_assets = sqlite_session.query(ViolationAsset).all()
        print(f"Found {len(sqlite_assets)} violation assets in SQLite.")
        for asset in sqlite_assets:
            existing = pg_session.query(ViolationAsset).filter_by(id=asset.id).first()
            if not existing:
                new_asset = ViolationAsset(
                    id=asset.id,
                    storage_backend=asset.storage_backend,
                    bucket_name=asset.bucket_name,
                    object_key=asset.object_key,
                    local_path=asset.local_path,
                    mime_type=asset.mime_type,
                    width=asset.width,
                    height=asset.height,
                    file_size_bytes=asset.file_size_bytes,
                    sha256=asset.sha256,
                    created_at=asset.created_at
                )
                pg_session.add(new_asset)
        pg_session.commit()
        print("ViolationAssets migrated successfully.")

        # 5. Migrate TrafficCounts
        print("Migrating TrafficCounts...")
        sqlite_counts = sqlite_session.query(TrafficCount).all()
        print(f"Found {len(sqlite_counts)} traffic counts in SQLite.")
        for count in sqlite_counts:
            existing = pg_session.query(TrafficCount).filter_by(id=count.id).first()
            if not existing:
                new_count = TrafficCount(
                    id=count.id,
                    camera_id=count.camera_id,
                    session_id=count.session_id,
                    timestamp=count.timestamp,
                    frame_id=count.frame_id,
                    vehicle_count=count.vehicle_count,
                    count_in=count.count_in,
                    count_out=count.count_out,
                    per_class_count=count.per_class_count,
                    latency_ms=count.latency_ms,
                    extra_data=count.extra_data
                )
                pg_session.add(new_count)
        pg_session.commit()
        print("TrafficCounts migrated successfully.")

        # 6. Migrate Violations
        print("Migrating Violations...")
        sqlite_violations = sqlite_session.query(Violation).all()
        print(f"Found {len(sqlite_violations)} violations in SQLite.")
        
        # Batch commit violations for performance
        batch_size = 100
        count = 0
        for violation in sqlite_violations:
            existing = pg_session.query(Violation).filter_by(id=violation.id).first()
            if not existing:
                new_violation = Violation(
                    id=violation.id,
                    camera_id=violation.camera_id,
                    session_id=violation.session_id,
                    traffic_count_id=violation.traffic_count_id,
                    asset_id=violation.asset_id,
                    track_id=violation.track_id,
                    timestamp=violation.timestamp,
                    violation_type=violation.violation_type,
                    confidence=violation.confidence,
                    class_name=violation.class_name,
                    frame_id=violation.frame_id,
                    bbox=violation.bbox,
                    image_path=violation.image_path,
                    reviewed=violation.reviewed,
                    severity=violation.severity,
                    notes=violation.notes,
                    extra_data=violation.extra_data
                )
                pg_session.add(new_violation)
                count += 1
                if count % batch_size == 0:
                    pg_session.commit()
                    print(f"  Migrated {count}/{len(sqlite_violations)} violations...")
        
        pg_session.commit()
        print(f"Violations migrated successfully (total {count} new added).")

        # 7. Update PostgreSQL auto-increment sequences
        print("Updating auto-increment sequences in PostgreSQL...")
        sequences = [
            ("users", "users_id_seq"),
            ("cameras", "cameras_id_seq"),
            ("processing_sessions", "processing_sessions_id_seq"),
            ("violation_assets", "violation_assets_id_seq"),
            ("traffic_counts", "traffic_counts_id_seq"),
            ("violations", "violations_id_seq")
        ]
        with pg_engine.connect() as conn:
            for table, seq in sequences:
                # Find maximum id
                max_id = conn.execute(text(f"SELECT COALESCE(MAX(id), 0) FROM {table}")).scalar()
                if max_id > 0:
                    conn.execute(text(f"SELECT setval('{seq}', {max_id})"))
                    print(f"  Reset {seq} to {max_id}")
            conn.commit()
            
        print("Migration completely successful!")
        
    except Exception as e:
        pg_session.rollback()
        print("Error during migration:", e)
        raise e
    finally:
        sqlite_session.close()
        pg_session.close()

if __name__ == "__main__":
    migrate()
