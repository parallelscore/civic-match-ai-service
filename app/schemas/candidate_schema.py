from typing import List, Union, Dict, Any, Optional
from app.schemas.to_camel_schema import CamelModel


class CandidateResponseItemSchema(CamelModel):
    """Schema for a single candidate's response to a question."""
    id: str
    question: str
    answer: Union[str, bool, List[str], Dict[str, Any]]
    comment: str
    election_id: str


class CandidateResponseSchema(CamelModel):
    """Schema for a candidate's responses with completion status and improved name handling."""
    candidate_id: str
    election_id: str
    responses: List[CandidateResponseItemSchema]

    # Completion status fields
    has_completed_profile: Optional[bool] = None
    has_completed_questionnaire: Optional[bool] = None

    # Optional fields that might be present in the data
    name: Optional[str] = None
    title: Optional[str] = None
    image_url: Optional[str] = None
    bio: Optional[str] = None

    def get_display_name(self) -> str:
        """Get the best available name for display."""
        if self.name:
            return self.name
        # You might want to add logic here to extract names from candidate_id
        # or fetch from another source
        return self.candidate_id

    def get_display_title(self) -> str:
        """Get the best available title for display."""
        if self.title:
            return self.title
        return "Candidate"

    def is_eligible_for_matching(self) -> bool:
        """
        Check if candidate is eligible for matching based on completion status.

        Returns:
            bool: True if candidate has completed profile and questionnaire and has responses
        """
        return (
                self.has_completed_profile is True and
                self.has_completed_questionnaire is True and
                len(self.responses) > 0
        )
