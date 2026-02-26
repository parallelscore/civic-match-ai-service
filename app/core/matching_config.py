# app/core/matching_config.py

from typing import Dict
from pydantic import BaseModel, Field


class MatchingConfiguration(BaseModel):
    """
    Single source of truth for every tunable value in the matching pipeline.
    Change a number here and it takes effect everywhere — no need to hunt
    through service files.
    """

    # ------------------------------------------------------------------ #
    # VOTER QUALITY GATE
    # A submission must have at least this many responses that contain
    # recognisable policy content before we attempt matching.
    # Responses that are clearly personal info (name, age, etc.) are
    # excluded from this count automatically.
    # ------------------------------------------------------------------ #
    min_policy_responses: int = Field(default=3, ge=1)

    # Minimum number of characters a FREE-TEXT answer must contain before we
    # treat it as potentially policy-relevant.
    # NOTE: This threshold is intentionally NOT applied to structured answers
    # (yes / no / agree / disagree / true / false etc.) — those are handled
    # by the context-aware LLM interpretation path regardless of length.
    # Boolean answers are always accepted regardless of this threshold.
    # Set to 4 so that "yes" (3) and "no" (2) still pass the quality gate
    # when they appear as plain text, while single-character noise is filtered.
    min_answer_length_for_text: int = Field(default=4)

    # ------------------------------------------------------------------ #
    # DIMENSION DISCOVERY
    # ------------------------------------------------------------------ #
    max_policy_dimensions: int = Field(default=6, ge=3, le=10)
    dimension_discovery_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # How many questions we sample when asking the LLM to discover topics.
    # Increase if elections have many unique question areas.
    dimension_discovery_question_sample: int = Field(default=30)

    # How many questions we send to the LLM per mapping batch.
    dimension_mapping_batch_size: int = Field(default=10)

    # ------------------------------------------------------------------ #
    # POSITION INFERENCE
    # ------------------------------------------------------------------ #
    position_inference_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # LLM blend ratio when a candidate provides a comment.
    # basic_weight + llm_weight should equal 1.0.
    position_basic_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    position_llm_weight: float = Field(default=0.7, ge=0.0, le=1.0)

    # Max tokens for position inference LLM calls.
    # Used for both the context-aware structured answer interpretation
    # and the comment enhancement path. 400 gives enough room for the
    # reasoning field without being wasteful.
    position_llm_max_tokens: int = Field(default=400)

    # Base confidence assigned before any adjustments.
    position_base_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    # Minimum comment length (chars) to treat as meaningful context.
    position_min_comment_length: int = Field(default=20)

    # ------------------------------------------------------------------ #
    # INTENSITY MULTIPLIERS
    # Maps answer phrases to how strongly held that position is.
    # 2.0 = very strong opinion, 1.0 = standard, 0.5 = mild/neutral
    # ------------------------------------------------------------------ #
    intensity_multipliers: Dict[str, float] = Field(default={
        "strongly agree": 2.0,
        "strongly support": 2.0,
        "strongly favor": 2.0,
        "strongly disagree": 2.0,
        "strongly oppose": 2.0,
        "strongly against": 2.0,
        "agree": 1.0,
        "support": 1.0,
        "favor": 1.0,
        "disagree": 1.0,
        "oppose": 1.0,
        "against": 1.0,
        "neutral": 0.5,
        "unsure": 0.5,
        "maybe": 0.5,
        "yes": 1.0,
        "no": 1.0,
    })

    # ------------------------------------------------------------------ #
    # MATCHING CALCULATOR
    # ------------------------------------------------------------------ #
    # Minimum number of policy dimensions that must overlap between voter
    # and candidate before we calculate a score.  Below this, the candidate
    # receives 0% rather than a misleading partial score.
    min_dimension_overlap: int = Field(default=2, ge=1)

    # When the position distance on the 0-100 scale is below this value,
    # apply a small alignment boost (rewards very close agreement).
    high_agreement_distance_threshold: float = Field(default=20.0)
    high_agreement_boost: float = Field(default=1.1)

    # How much of the voter values profile to surface (top N dimensions).
    voter_profile_max_items: int = Field(default=6)

    # How far from neutral (50/100) a position score must be before we
    # consider it a meaningful opinion worth surfacing in the values profile.
    # Lowered to 10.0 now that context-aware LLM scoring produces more
    # precise, meaningful scores even at smaller deviations from centre.
    voter_profile_neutrality_threshold: float = Field(default=10.0)

    # Top N candidates assigned "TOP" category.
    top_candidate_count: int = Field(default=3)

    # ------------------------------------------------------------------ #
    # CONSISTENCY ANALYSIS
    # ------------------------------------------------------------------ #
    enable_consistency_analysis: bool = Field(default=True)
    # Maximum fraction of the match score that can be deducted for
    # logical inconsistencies. Reduced to 0.15 now that tension detection
    # is more nuanced — a 20% cap was too aggressive for minor tensions.
    consistency_penalty_rate: float = Field(default=0.15, ge=0.0, le=0.5)

    # Severity → penalty weight mapping.
    consistency_severity_weights: Dict[str, float] = Field(default={
        "low": 0.1,
        "medium": 0.3,
        "high": 0.5,
    })

    # ------------------------------------------------------------------ #
    # CONFIDENCE WEIGHTING
    # ------------------------------------------------------------------ #
    confidence_weighting_enabled: bool = Field(default=True)
    # Positions with confidence below this value are excluded from matching.
    # 0.3 ensures only positions the system is reasonably certain about
    # are used — very uncertain inferences do not pollute the final score.
    min_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # ------------------------------------------------------------------ #
    # LLM CALL LIMITS
    # ------------------------------------------------------------------ #
    llm_max_tokens_dimension_discovery: int = Field(default=2000)
    llm_max_tokens_question_mapping: int = Field(default=1500)
    # 150 tokens gives descriptions enough room to be informative without
    # being cut off mid-sentence.
    llm_max_tokens_voter_value_description: int = Field(default=150)
    llm_max_tokens_position_description: int = Field(default=150)
    llm_temperature_voter_description: float = Field(default=0.4)
    llm_temperature_position_description: float = Field(default=0.3)

    # How many voter responses the legacy generate_voter_profile LLM call
    # is allowed to see (kept for backward compat, not used in V2 path).
    llm_voter_profile_response_limit: int = Field(default=8)

    # ------------------------------------------------------------------ #
    # CACHING
    # ------------------------------------------------------------------ #
    cache_dimension_discovery: bool = Field(default=True)
    cache_position_inference: bool = Field(default=True)
    dimension_cache_ttl: int = Field(default=86400)   # 24 hours
    position_cache_ttl: int = Field(default=3600)     # 1 hour
    description_cache_ttl: int = Field(default=3600)  # 1 hour

    # ------------------------------------------------------------------ #
    # RETRY / TIMEOUT
    # ------------------------------------------------------------------ #
    max_retries: int = Field(default=3)
    request_timeout: int = Field(default=60)


# Single global instance — import this everywhere instead of magic numbers.
matching_config = MatchingConfiguration()

