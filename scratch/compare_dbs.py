from sqlalchemy import create_engine, text

pg_url = "postgresql://traffic_ai_user:26udsfygkrrX7ecUiZN9JrI1ET90SVri@dpg-d7p26ecm0tmc73dgfdig-a.singapore-postgres.render.com/traffic_ai"
sqlite_url = "sqlite:///./traffic_system.db"

def get_version(url, name):
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            result = conn.execute(text("select version_num from alembic_version"))
            versions = [row[0] for row in result]
            print(f"{name} version:", versions)
            
            # Count users and cameras
            users_count = conn.execute(text("select count(*) from users")).scalar()
            cameras_count = conn.execute(text("select count(*) from cameras")).scalar()
            violations_count = conn.execute(text("select count(*) from violations")).scalar()
            print(f"  Users: {users_count}, Cameras: {cameras_count}, Violations: {violations_count}")
    except Exception as e:
        print(f"Error reading from {name}: {e}")

get_version(sqlite_url, "SQLite")
get_version(pg_url, "Cloud PostgreSQL")
