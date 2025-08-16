from pydantic import BaseModel, Field
from typing import List

class TranscribeResponse(BaseModel):
    transcript: str

class SummariseRequest(BaseModel):
    transcript: str = Field(..., description="Raw transcript text")

class SummariseResponse(BaseModel):
    topics: List[str]
    topic_weights: List[float]
    summary: str

class MatchRequest(BaseModel):
    user_id_1: str
    user_id_2: str
    topics: List[str]
    topic_weights: List[float]
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="weight for psychometrics")
    beta: float = Field(0.5, ge=0.0, le=1.0, description="weight for topic vector")

class MatchResponse(BaseModel):
    score: float
    label: str
    detail: str