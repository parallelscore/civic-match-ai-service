from sqlalchemy import Column, String, Boolean, DateTime, Text

from app.api.models.base_model import BaseModel
from app.utils.postgresql_db_util import db_util

base = db_util.base


class ElectionModel(base, BaseModel):
    """Model representing an election or matching campaign."""
    __tablename__ = "elections"
    __table_args__ = ({"schema": 'public'})

    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    location = Column(String, nullable=True)  # City, district, etc.
    election_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<ElectionModel(id='{self.id}', name='{self.name}')>"
