from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Optional


class VoterResponseItemSchema(BaseModel):
    """Schema for a voter's response to a question."""
    question_id: str
    question: str
    answer: Union[str, bool, List[str], Dict[str, Any]]
    category: Optional[str] = None


class VoterSubmissionSchema(BaseModel):
    """Schema for a voter's submission of responses."""
    election_id: str
    citizen_id: str
    responses: List[VoterResponseItemSchema]
    completed_at: datetime = Field(default_factory=datetime.now)


class IssueMatchDetailSchema(BaseModel):
    """Schema for detailed position comparison on a single issue."""
    issue: str
    alignment: str  # "Strongly Aligned", "Moderately Aligned", "Weakly Aligned"
    voter_position: str
    candidate_position: str


class CandidateMatchSchema(BaseModel):
    """Schema for a match result between a voter and a candidate."""
    candidate_id: str
    candidate_name: str
    candidate_title: str
    match_percentage: int  # 0-100
    top_aligned_issues: List[str]
    issue_matches: List[IssueMatchDetailSchema]


class MatchResultsResponseSchema(BaseModel):
    """Schema for the response containing all match results."""
    voter_id: str
    election_id: str
    matches: List[CandidateMatchSchema]
    generated_at: datetime = Field(default_factory=datetime.now)
