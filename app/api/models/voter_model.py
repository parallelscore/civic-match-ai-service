from sqlalchemy import Column, String, ForeignKey, JSON, DateTime

from app.api.models.base_model import BaseModel
from app.api.models.election_model import ElectionModel
from app.api.models.question_model import QuestionModel
from app.utils.postgresql_db_util import db_util

base = db_util.base


class VoterModel(base, BaseModel):
    """Model representing a voter/user in the system."""
    __tablename__ = "voters"
    __table_args__ = {"schema": 'public'}

    citizen_id = Column(String, nullable=False, unique=True)  # External ID from the frontend
    demographic_info = Column(JSON, nullable=True)  # Optional demographic information

    def __repr__(self):
        return f"<VoterModel(id='{self.id}', citizen_id='{self.citizen_id}')>"


class VoterResponseModel(base, BaseModel):
    """Model representing a voter's response to a question."""
    __tablename__ = "voter_responses"
    __table_args__ = {"schema": 'public'}

    voter_id = Column(String, ForeignKey(VoterModel.id), nullable=False)
    question_id = Column(String, ForeignKey(QuestionModel.id), nullable=False)
    election_id = Column(String, ForeignKey(ElectionModel.id), nullable=False)
    answer = Column(JSON, nullable=False)  # Store any response type as JSON
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<VoterResponseModel(voter='{self.voter_id}', question='{self.question_id}')>"
