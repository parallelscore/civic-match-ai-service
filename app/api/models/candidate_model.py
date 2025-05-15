from sqlalchemy import Column, String, Text, ForeignKey, JSON

from app.api.models.base_model import BaseModel
from app.api.models.election_model import ElectionModel
from app.api.models.question_model import QuestionModel
from app.utils.postgresql_db_util import db_util

base = db_util.base


class CandidateModel(base, BaseModel):
    """Model representing a candidate in an election."""
    __tablename__ = "candidates"
    __table_args__ = ({"schema": 'public'})

    name = Column(String, nullable=False)
    bio = Column(Text, nullable=True)
    election_id = Column(String, ForeignKey(ElectionModel.id), nullable=False)
    profile_data = Column(JSON, nullable=True)  # Additional profile information

    def __repr__(self):
        return f"<CandidateModel(id='{self.id}', name='{self.name}')>"


class CandidateResponseModel(base, BaseModel):
    """Model representing a candidate's response to a question."""
    __tablename__ = "candidate_responses"
    __table_args__ = {"schema": 'public'}

    candidate_id = Column(String, ForeignKey(CandidateModel.id), nullable=False)
    question_id = Column(String, ForeignKey(QuestionModel.id), nullable=False)
    answer = Column(JSON, nullable=False)  # Store any response type as JSON
    comment = Column(Text, nullable=True)  # Additional explanation or comment

    def __repr__(self):
        return f"<CandidateResponseModel(candidate='{self.candidate_id}', question='{self.question_id}')>"
