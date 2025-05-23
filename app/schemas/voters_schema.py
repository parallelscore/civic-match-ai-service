from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Optional

from app.schemas.to_camel_schema import CamelModel


class VoterResponseItemSchema(CamelModel):
    """Schema for a voter's response to a question."""
    question_id: str
    question: str
    answer: Union[str, bool, List[str], Dict[str, Any]]
    category: Optional[str] = None


class VoterSubmissionSchema(CamelModel):
    """Schema for a voter's submission of responses."""
    election_id: str
    citizen_id: str
    responses: List[VoterResponseItemSchema]
    completed_at: datetime = Field(default_factory=datetime.now)


class VoterValueProfileSchema(BaseModel):
    """Schema for voter's political values profile."""
    issue: str
    description: str
    priority_level: str  # "High", "Medium", "Low"


class IssueMatchDetailSchema(BaseModel):
    """Schema for detailed position comparison on a single issue."""
    issue: str
    alignment: str  # "Strongly Aligned", "Moderately Aligned", "Weakly Aligned"
    alignment_score: float  # 0.0 to 1.0
    voter_position: str
    candidate_position: str
    explanation: Optional[str] = None  # LLM-generated explanation of why they align/don't align


class TopAlignedIssueSchema(BaseModel):
    """Schema for top aligned issues tags."""
    issue: str
    color: str  # For UI styling - e.g., "orange", "blue", "green"


class CandidateMatchSchema(BaseModel):
    """Schema for a match result between a voter and a candidate."""
    candidate_id: str
    candidate_name: str
    candidate_title: str
    candidate_image_url: Optional[str] = None
    match_percentage: int  # 0-100
    match_strength_visual: float  # 0.0 to 1.0 for progress bar
    top_aligned_issues: List[TopAlignedIssueSchema]
    issue_matches: List[IssueMatchDetailSchema]
    overall_explanation: Optional[str] = None  # LLM-generated summary of the match


class MatchResultsResponseSchema(BaseModel):
    """Schema for the response containing all match results."""
    citizen_id: str
    election_id: str
    voter_values_profile: List[VoterValueProfileSchema]
    matches: List[CandidateMatchSchema]
    generated_at: datetime = Field(default_factory=datetime.now)
    processing_method: str  # "direct", "semantic", "llm_enhanced", "hybrid"
    confidence_score: float  # Overall confidence in the matching results


class QuestionTopicSchema(BaseModel):
    """Schema for LLM-discovered topics."""
    topic_id: str
    topic_name: str
    topic_description: str
    questions: List[str]
    importance_weight: float = 1.0


class ElectionTopicsSchema(BaseModel):
    """Schema for all topics discovered in an election."""
    election_id: str
    topics: List[QuestionTopicSchema]
    discovered_at: datetime = Field(default_factory=datetime.now)
    discovery_method: str  # "llm", "clustering", "manual"


class QuestionSimilaritySchema(BaseModel):
    """Schema for question similarity results."""
    voter_question: str
    candidate_question: str
    similarity_score: float
    similarity_method: str  # "embedding", "llm", "hybrid"
    explanation: Optional[str] = None