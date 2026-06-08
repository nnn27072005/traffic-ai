import os
from sqlalchemy import create_engine, text

db_url = "postgresql://traffic_ai_user:26udsfygkrrX7ecUiZN9JrI1ET90SVri@dpg-d7p26ecm0tmc73dgfdig-a.singapore-postgres.render.com/traffic_ai"
print("Connecting to:", db_url)

try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        print("Successfully connected to the database!")
        # Check tables
        result = conn.execute(text("select tablename from pg_tables where schemaname='public'"))
        tables = [row[0] for row in result]
        print("Existing tables:", tables)
except Exception as e:
    print("Error connecting to database:", e)
