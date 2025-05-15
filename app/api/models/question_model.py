from sqlalchemy import Column, String, Boolean, Float, Text, ForeignKey, JSON

from app.api.models.base_model import BaseModel
from app.api.models.election_model import ElectionModel
from app.utils.postgresql_db_util import db_util

base = db_util.base


class QuestionModel(base, BaseModel):
    """Model representing a question or issue in the survey."""
    __tablename__ = "questions"
    __table_args__ = ({"schema": 'public'})

    text = Column(Text, nullable=False)
    category = Column(String, nullable=True)
    response_type = Column(String, nullable=False)  # binary, multiple-choice, ranking, text
    weight = Column(Float, default=1.0)
    election_id = Column(String, ForeignKey(ElectionModel.id), nullable=False)
    is_active = Column(Boolean, default=True)
    metadatas = Column(JSON, nullable=True)  # Store additional config like choices for multiple-choice

    def __repr__(self):
        return f"<QuestionModel(id='{self.id}', text='{self.text[:30]}...', type='{self.response_type}')>"
