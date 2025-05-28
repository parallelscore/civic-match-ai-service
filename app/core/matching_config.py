# app/core/matching_config.py

from typing import Dict
from pydantic import BaseModel, Field

class MatchingConfiguration(BaseModel):
    """Centralized configuration for the enhanced matching system"""

    # Dimension Discovery
    max_policy_dimensions: int = Field(default=6, ge=3, le=10)
    dimension_discovery_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # Intensity Multipliers
    intensity_multipliers: Dict[str, float] = Field(default={
        "strongly_agree": 2.0,
        "strongly_support": 2.0,
        "agree": 1.0,
        "support": 1.0,
        "neutral": 0.5,
        "disagree": 1.0,
        "oppose": 1.0,
        "strongly_disagree": 2.0,
        "strongly_oppose": 2.0,
        "yes": 1.0,
        "no": 1.0
    })

    # Consistency Analysis
    consistency_penalty_rate: float = Field(default=0.20, ge=0.0, le=0.5)
    enable_consistency_analysis: bool = Field(default=True)

    # Confidence Weighting
    confidence_weighting_enabled: bool = Field(default=True)
    min_confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # Position Inference
    position_inference_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # Caching
    cache_dimension_discovery: bool = Field(default=True)
    cache_position_inference: bool = Field(default=True)
    dimension_cache_ttl: int = Field(default=86400)  # 24 hours

    # LLM Settings
    max_retries: int = Field(default=3)
    request_timeout: int = Field(default=60)

# Global config instance
matching_config = MatchingConfiguration()
