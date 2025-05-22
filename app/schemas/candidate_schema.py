from typing import List, Union, Dict, Any
from app.schemas.to_camel_schema import CamelModel


class CandidateResponseItemSchema(CamelModel):
    """Schema for a single candidate's response to a question."""
    id: str
    question: str
    answer: Union[str, bool, List[str], Dict[str, Any]]
    comment: str
    election_id: str


class CandidateResponseSchema(CamelModel):
    """Schema for a candidate's responses."""
    candidate_id: str
    election_id: str
    responses: List[CandidateResponseItemSchema]
