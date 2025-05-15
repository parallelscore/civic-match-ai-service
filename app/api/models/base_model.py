import uuid

from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func


class BaseModel:
    """Base model that provides common fields and functionality for all models."""

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def to_dict(self):
        """Convert the model instance to a dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
