"""Initial schema — 5 core tables

Revision ID: 001
Revises: None
Create Date: 2026-04-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── cameras ───────────────────────────────────────────────────
    op.create_table(
        "cameras",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("location", sa.String(255), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("stream_url", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── processing_sessions ──────────────────────────────────────
    op.create_table(
        "processing_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("camera_id", sa.Integer(), sa.ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("source_type", sa.String(32), nullable=False, server_default="upload",
                  comment="upload | stream | rtsp"),
        sa.Column("source_path", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending", index=True,
                  comment="pending | processing | completed | failed"),
        sa.Column("total_frames", sa.Integer(), nullable=True),
        sa.Column("processed_frames", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ── violation_assets ─────────────────────────────────────────
    op.create_table(
        "violation_assets",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("storage_backend", sa.String(32), nullable=False, server_default="local"),
        sa.Column("bucket_name", sa.String(255), nullable=True),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("local_path", sa.Text(), nullable=True),
        sa.Column("mime_type", sa.String(128), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=True, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── traffic_counts ───────────────────────────────────────────
    op.create_table(
        "traffic_counts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("camera_id", sa.Integer(), sa.ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("processing_sessions.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column("frame_id", sa.Integer(), nullable=True, index=True),
        sa.Column("vehicle_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("count_in", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("count_out", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("per_class_count", sa.JSON(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("metadata", sa.JSON(), nullable=True),
    )

    # ── violations ───────────────────────────────────────────────
    op.create_table(
        "violations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("camera_id", sa.Integer(), sa.ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("processing_sessions.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("traffic_count_id", sa.Integer(), sa.ForeignKey("traffic_counts.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("asset_id", sa.Integer(), sa.ForeignKey("violation_assets.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("track_id", sa.Integer(), nullable=False, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column("violation_type", sa.String(64), nullable=False, index=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("class_name", sa.String(64), nullable=False, server_default="'unknown'"),
        sa.Column("frame_id", sa.Integer(), nullable=True, index=True),
        sa.Column("bbox", sa.JSON(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=True),
        sa.Column("reviewed", sa.Boolean(), nullable=False, server_default=sa.text("false"), index=True),
        sa.Column("severity", sa.String(16), nullable=False, server_default="'medium'", index=True,
                  comment="low | medium | high | critical"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.UniqueConstraint("camera_id", "track_id", "frame_id", "violation_type", name="uq_violation_event"),
    )


def downgrade() -> None:
    op.drop_table("violations")
    op.drop_table("traffic_counts")
    op.drop_table("violation_assets")
    op.drop_table("processing_sessions")
    op.drop_table("cameras")
