# app/models.py
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


# -------- Transcribe --------
class TranscribeResponse(BaseModel):
    text: str = Field(..., description="Transcript text extracted from audio")


# -------- Summarise --------
class SummariseRequest(BaseModel):
    transcript: str
    top_k: int = 5


class SummariseResponse(BaseModel):
    topics: List[str]
    summary: str
    # 384-d vector from SentenceTransformers (or whatever your embedding dim is)
    topic_embedding: List[float]


# -------- Match --------
class MatchRequest(BaseModel):
    # We no longer ask for user ids; they are auto-picked from your synthetic JSON.
    topics: List[str] = Field(..., description="Topic phrases to embed & use for matching")
    # Per-user topic affinity scaling (your requested '80% vs 60%' style)
    topic_scale_user1: float = Field(1.0, ge=0.0, description="Scale applied to topic vector for user1")
    topic_scale_user2: float = Field(1.0, ge=0.0, description="Scale applied to topic vector for user2")


class MatchResponse(BaseModel):
    score: float
    label: str
    detail: str
