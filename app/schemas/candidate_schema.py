from typing import List, Union, Dict, Any

from pydantic import BaseModel


class CandidateResponseItemSchema(BaseModel):
    """Schema for a single candidate's response to a question."""
    id: str
    question: str
    answer: Union[str, bool, List[str], Dict[str, Any]]
    comment: str
    election_id: str


class CandidateResponseSchema(BaseModel):
    """Schema for a candidate's responses."""
    candidate_id: str
    election_id: str
    responses: List[CandidateResponseItemSchema]
