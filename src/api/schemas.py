"""Pydantic schemas for API request/response serialization.

Organized into sections:
  - Shared (BBox, Track, Violation event)
  - Analysis pipeline (AnalysisResult)
  - REST API responses (ViolationResponse, SessionResponse, etc.)
  - Health & Benchmark
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════
#  SHARED SCHEMAS
# ══════════════════════════════════════════════════════════════════


class BBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class TrackSchema(BaseModel):
    track_id: int
    class_name: str
    confidence: float
    bbox: BBoxSchema


class ViolationSchema(BaseModel):
    track_id: int
    violations: list[str]
    confidence: float
    bbox: BBoxSchema


# ══════════════════════════════════════════════════════════════════
#  ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════


class AnalysisResult(BaseModel):
    frame_id: int
    vehicle_count: int
    tracks: list[TrackSchema]
    violations: list[ViolationSchema]
    count_in: int
    count_out: int
    latency_ms: float


# ══════════════════════════════════════════════════════════════════
#  REST API — CAMERAS
# ══════════════════════════════════════════════════════════════════


class CameraCreate(BaseModel):
    name: str
    location: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    stream_url: str | None = None
    description: str | None = None
    is_active: bool = True


class CameraResponse(BaseModel):
    id: int
    name: str
    location: str | None
    latitude: float | None
    longitude: float | None
    stream_url: str | None
    description: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ══════════════════════════════════════════════════════════════════
#  REST API — PROCESSING SESSIONS
# ══════════════════════════════════════════════════════════════════


class SessionCreate(BaseModel):
    camera_id: int | None = None
    source_type: str = "upload"
    source_path: str | None = None
    config: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    id: int
    camera_id: int | None
    source_type: str
    source_path: str | None
    status: str
    total_frames: int | None
    processed_frames: int | None
    fps: float | None
    error_message: str | None
    config: dict[str, Any] | None
    started_at: datetime
    completed_at: datetime | None
    output_video_path: str | None
    total_violations: int | None = 0

    class Config:
        from_attributes = True


class SessionUpdate(BaseModel):
    status: str | None = None
    processed_frames: int | None = None
    total_frames: int | None = None
    fps: float | None = None
    error_message: str | None = None
    completed_at: datetime | None = None


# ══════════════════════════════════════════════════════════════════
#  REST API — VIOLATIONS
# ══════════════════════════════════════════════════════════════════


class ViolationRecordSchema(BaseModel):
    """Detailed violation response with asset info."""

    id: int
    camera_id: int | None = None
    session_id: int | None = None
    track_id: int
    timestamp: datetime
    violation_type: str
    confidence: float
    class_name: str
    frame_id: int | None = None
    bbox: dict
    image_path: str | None = None
    reviewed: bool = False
    severity: str = "medium"
    notes: str | None = None
    asset_url: str | None = None

    class Config:
        from_attributes = True


class ViolationReview(BaseModel):
    """Payload for marking a violation as reviewed."""

    reviewed: bool = True
    severity: str | None = None
    notes: str | None = None


class ViolationFilter(BaseModel):
    """Query parameters for filtering violations."""

    camera_id: int | None = None
    session_id: int | None = None
    violation_type: str | None = None
    severity: str | None = None
    reviewed: bool | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


# ══════════════════════════════════════════════════════════════════
#  REST API — STATS / SUMMARY
# ══════════════════════════════════════════════════════════════════


class SummaryStats(BaseModel):
    total_vehicles: int
    total_violations: int
    avg_latency: float
    count_in: int
    count_out: int
    violations_by_severity: dict[str, int] = Field(default_factory=dict)
    violations_by_type: dict[str, int] = Field(default_factory=dict)


class TrafficTimeSeriesPoint(BaseModel):
    timestamp: datetime
    vehicle_count: int
    count_in: int
    count_out: int
    per_class_count: dict[str, int] | None = None


class TrafficTimeSeries(BaseModel):
    points: list[TrafficTimeSeriesPoint]
    total_records: int


# ══════════════════════════════════════════════════════════════════
#  HEALTH & BENCHMARK
# ══════════════════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    status: str
    traffic_model: str
    helmet_model: str
    device: str


class BenchmarkRow(BaseModel):
    name: str
    mean_ms: float
    fps: float


class BenchmarkResponse(BaseModel):
    gpu: str
    results: list[BenchmarkRow]


# ══════════════════════════════════════════════════════════════════
#  AUTHENTICATION
# ══════════════════════════════════════════════════════════════════


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: str | None = None


class UserResponse(BaseModel):
    id: int
    username: str
    full_name: str | None
    email: str | None
    role: str
    avatar_url: str | None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


class TokenData(BaseModel):
    username: str | None = None


class GoogleLoginRequest(BaseModel):
    token: str


class VideoUrlRequest(BaseModel):
    url: str
    stream_mode: bool = True
    save_output: bool = False
    save_evidence: bool = False
    analysis_fps: float | None = None
