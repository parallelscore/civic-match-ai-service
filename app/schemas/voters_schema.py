from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Optional


class VoterResponseItemSchema(BaseModel):
    """Schema for a voter's response to a question."""
    question_id: str
    question: str
    answer: Union[str, bool, List[str], Dict[str, Any]]
    category: Optional[str] = None


class IssuePrioritySchema(BaseModel):
    """Schema for a voter's issue priorities."""
    category: str
    weight: float  # 0.0-1.0, where 1.0 is the highest priority


class VoterSubmissionSchema(BaseModel):
    """Schema for a voter's submission of responses."""
    election_id: str
    citizen_id: str
    responses: List[VoterResponseItemSchema]
    completed_at: datetime = Field(default_factory=datetime.now)
    issue_priorities: Optional[List[IssuePrioritySchema]] = None


class IssueMatchDetailSchema(BaseModel):
    """Schema for detailed position comparison on a single issue."""
    issue: str
    alignment: str  # "Strongly Aligned", "Moderately Aligned", "Weakly Aligned"
    voter_position: str
    candidate_position: str


class WeightedIssueSchema(BaseModel):
    """Schema for an issue with its weight in the match calculation."""
    category: str
    weight: float
    impact_score: float  # How much this issue affected the match (0.0-1.0)


class CandidateMatchSchema(BaseModel):
    """Schema for a match result between a voter and a candidate."""
    candidate_id: str
    candidate_name: str
    candidate_title: str
    match_percentage: int  # 0-100
    confidence_score: float = 1.0
    top_aligned_issues: List[str]
    issue_matches: List[IssueMatchDetailSchema]
    weighted_issues: Optional[List[WeightedIssueSchema]]  # New field showing priority impact
    strongest_match_factors: Optional[List[str]]


class MatchResultsResponseSchema(BaseModel):
    """Schema for the response containing all match results."""
    voter_id: str
    election_id: str
    matches: List[CandidateMatchSchema]
    generated_at: datetime = Field(default_factory=datetime.now)
