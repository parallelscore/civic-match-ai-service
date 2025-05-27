# app/schemas/policy_matching_schema.py
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

class TensionSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class PolicyDimension(BaseModel):
    """Represents a discovered policy dimension"""
    dimension_id: str
    name: str
    description: str
    policy_spectrum_description: str  # e.g., "Opposition (0) to Strong Support (100)"
    keywords: List[str]
    confidence: float = Field(ge=0.0, le=1.0)

class QuestionDimensionMapping(BaseModel):
    """Maps questions to policy dimensions"""
    question: str
    primary_dimension_id: str
    secondary_dimension_ids: List[str] = []
    primary_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    secondary_weights: Dict[str, float] = {}
    mapping_confidence: float = Field(ge=0.0, le=1.0)

class PolicyPosition(BaseModel):
    """Represents a person's position on a policy dimension"""
    dimension_id: str
    position_score: float = Field(ge=0.0, le=100.0)  # 0-100 scale
    confidence: float = Field(ge=0.0, le=1.0)
    intensity_multiplier: float = Field(ge=0.0, le=3.0)
    reasoning: str = ""
    source_question: str = ""
    source_answer: Any = None

class LogicalTension(BaseModel):
    """Represents inconsistency between policy positions"""
    dimension_1_id: str
    dimension_2_id: str
    tension_type: str
    severity: TensionSeverity
    impact_score: float = Field(ge=0.0, le=1.0)
    explanation: str

class PersonPolicyProfile(BaseModel):
    """Complete policy profile for a person (voter or candidate)"""
    person_id: str
    person_type: str  # "voter" or "candidate"
    policy_positions: List[PolicyPosition]
    logical_tensions: List[LogicalTension]
    overall_consistency_score: float = Field(ge=0.0, le=1.0)

class DimensionMatch(BaseModel):
    """Match result for a single policy dimension"""
    dimension_id: str
    dimension_name: str
    alignment_score: float = Field(ge=0.0, le=1.0)
    confidence_weighted_score: float = Field(ge=0.0, le=1.0)
    voter_position: PolicyPosition
    candidate_position: PolicyPosition
    alignment_explanation: str
    voter_position_description: Optional[str] = None      # Add this
    candidate_position_description: Optional[str] = None  # Add this

class EnhancedMatchResult(BaseModel):
    """Complete match result between voter and candidate"""
    voter_id: str
    candidate_id: str
    overall_match_percentage: int = Field(ge=0, le=100)
    confidence_weighted_percentage: int = Field(ge=0, le=100)
    dimension_matches: List[DimensionMatch]
    consistency_penalty_applied: float = Field(ge=0.0, le=1.0)
    match_explanation: str
    top_aligned_dimensions: List[str]

class ElectionPolicyAnalysis(BaseModel):
    """Complete policy analysis for an election"""
    election_id: str
    discovered_dimensions: List[PolicyDimension]
    question_mappings: List[QuestionDimensionMapping]
    discovery_confidence: float = Field(ge=0.0, le=1.0)
    analysis_timestamp: str