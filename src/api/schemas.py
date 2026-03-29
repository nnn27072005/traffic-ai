# src/api/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field


class BBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class TrackSchema(BaseModel):
    track_id:   int
    class_name: str
    confidence: float
    bbox:       BBoxSchema


class ViolationSchema(BaseModel):
    track_id:   int
    violations: list[str]
    confidence: float
    bbox:       BBoxSchema


class AnalysisResult(BaseModel):
    frame_id:       int
    vehicle_count:  int
    tracks:         list[TrackSchema]
    violations:     list[ViolationSchema]
    count_in:       int
    count_out:      int
    latency_ms:     float


class HealthResponse(BaseModel):
    status:         str
    traffic_model:  str
    helmet_model:   str
    device:         str


class BenchmarkRow(BaseModel):
    name:     str
    mean_ms:  float
    fps:      float


class BenchmarkResponse(BaseModel):
    gpu:     str
    results: list[BenchmarkRow]