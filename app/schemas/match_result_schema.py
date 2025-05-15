from datetime import datetime
from typing import List, Dict, Any

from pydantic import BaseModel, Field


class MatchResultDetailSchema(BaseModel):
    """Schema for detailed match result information."""
    category: str
    score: float
    alignment: str  # high, medium, low
    key_issues: List[Dict[str, Any]]


class MatchResultSchema(BaseModel):
    """Schema for a match result between a voter and a candidate."""
    candidate_id: str
    overall_score: float
    details: List[MatchResultDetailSchema]
    top_aligned_issues: List[str]
    top_misaligned_issues: List[str]


class MatchResultsResponseSchema(BaseModel):
    """Schema for the response containing all match results."""
    voter_id: str
    election_id: str
    matches: List[MatchResultSchema]
    generated_at: datetime = Field(default_factory=datetime.now)
