from datetime import datetime
from typing import List, Union, Dict, Any

from pydantic import BaseModel, Field


class VoterResponseItemSchema(BaseModel):
    """Schema for a voter's response to a question."""
    question_id: str
    question: str
    answer: Union[str, bool, List[str], Dict[str, Any]]


class VoterSubmissionSchema(BaseModel):
    """Schema for a voter's submission of responses."""
    election_id: str
    citizen_id: str
    responses: List[VoterResponseItemSchema]
    completed_at: datetime = Field(default_factory=datetime.now)
