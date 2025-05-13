from sqlalchemy import Column, String, Float, ForeignKey, JSON, DateTime
from sqlalchemy.sql import func

from app.api.models.base_model import BaseModel
from app.api.models.candidate_model import CandidateModel
from app.api.models.election_model import ElectionModel
from app.api.models.voter_model import VoterModel
from app.utils.postgresql_db_util import db_util

base = db_util.base


class MatchResultModel(base, BaseModel):
    """Model representing a match result between a voter and a candidate."""
    __tablename__ = "match_results"
    __table_args__ = {"schema": 'public'}

    voter_id = Column(String, ForeignKey(VoterModel.id), nullable=False)
    candidate_id = Column(String, ForeignKey(CandidateModel.id), nullable=False)
    election_id = Column(String, ForeignKey(ElectionModel.id), nullable=False)
    overall_score = Column(Float, nullable=False)
    issue_scores = Column(JSON, nullable=True)  # Detailed scores by issue/category
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    explanation = Column(JSON, nullable=True)  # Explanation for the match

    def __repr__(self):
        return f"<MatchResultModel(voter='{self.voter_id}', candidate='{self.candidate_id}', score='{self.overall_score}')>"


class UserFeedbackModel(base, BaseModel):
    """Model representing user feedback on match results."""
    __tablename__ = "user_feedback"
    __table_args__ = {"schema": 'public'}

    match_result_id = Column(String, ForeignKey(MatchResultModel.id), nullable=False)
    rating = Column(Float, nullable=True)  # User rating of the match quality
    comments = Column(String, nullable=True)  # User comments on the match

    def __repr__(self):
        return f"<UserFeedbackModel(match='{self.match_result_id}', rating='{self.rating}')>"
