# tests/test_implementation_details.py
#
# V2 implementation detail tests.
# Verifies internal service logic that underpins the matching pipeline:
#   - Position inference (context-aware, structured, free-text)
#   - Consistency analyzer (generic domain-agnostic tension detection)
#   - Enhanced matching calculator (overlap guard, confidence filtering)
#   - Policy dimension discovery (generic fallbacks)
#   - Matching config (all values present and sane)

import pytest
from unittest.mock import patch, AsyncMock

from app.core.matching_config import matching_config
from app.schemas.policy_matching_schema import (
    PolicyDimension,
    PolicyPosition,
    TensionSeverity,
)
from app.services.position_inference_service import PositionInferenceService
from app.services.consistency_analyzer_service import ConsistencyAnalyzerService
from app.services.enhanced_matching_calculator_service import EnhancedMatchingCalculatorService
from app.services.policy_dimension_discovery_service import PolicyDimensionDiscoveryService


# ── Shared Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def education_dimension():
    return PolicyDimension(
        dimension_id="education_funding",
        name="Education Funding",
        description="Policies related to education investment and budget allocation",
        policy_spectrum_description="Reduce Funding (0) to Increase Funding (100)",
        keywords=["education", "school", "funding", "budget"],
        confidence=0.9,
    )


@pytest.fixture
def tax_dimension():
    return PolicyDimension(
        dimension_id="tax_revenue",
        name="Tax & Revenue Policy",
        description="Policies about taxation and public revenue generation",
        policy_spectrum_description="Lower Taxes (0) to Higher Taxes (100)",
        keywords=["tax", "revenue", "levy", "fiscal"],
        confidence=0.9,
    )


@pytest.fixture
def healthcare_dimension():
    return PolicyDimension(
        dimension_id="healthcare_funding",
        name="Healthcare Funding",
        description="Policies about public healthcare investment",
        policy_spectrum_description="Reduce Services (0) to Expand Services (100)",
        keywords=["healthcare", "health", "fund", "service"],
        confidence=0.9,
    )


@pytest.fixture
def position_inference_service():
    return PositionInferenceService()


@pytest.fixture
def consistency_analyzer():
    return ConsistencyAnalyzerService()


@pytest.fixture
def calculator():
    return EnhancedMatchingCalculatorService()


@pytest.fixture
def discovery_service():
    return PolicyDimensionDiscoveryService()


# ── Matching Config Tests ─────────────────────────────────────────────────────

class TestMatchingConfig:
    """Verify all V2 config fields exist and have sensible defaults."""

    def test_voter_quality_gate_config_present(self):
        assert matching_config.min_policy_responses >= 1
        assert matching_config.min_answer_length_for_text >= 1

    def test_dimension_discovery_config_present(self):
        assert matching_config.max_policy_dimensions >= 3
        assert matching_config.dimension_discovery_question_sample >= 10
        assert matching_config.dimension_mapping_batch_size >= 5

    def test_position_inference_config_present(self):
        assert 0.0 <= matching_config.position_basic_weight <= 1.0
        assert 0.0 <= matching_config.position_llm_weight <= 1.0
        assert abs(
            matching_config.position_basic_weight + matching_config.position_llm_weight - 1.0
        ) < 0.001  # Should sum to 1.0
        assert matching_config.position_llm_max_tokens >= 100
        assert 0.0 <= matching_config.position_base_confidence <= 1.0

    def test_matching_calculator_config_present(self):
        assert matching_config.min_dimension_overlap >= 1
        assert matching_config.high_agreement_distance_threshold > 0
        assert matching_config.high_agreement_boost >= 1.0
        assert matching_config.voter_profile_max_items >= 1
        assert matching_config.voter_profile_neutrality_threshold > 0
        assert matching_config.top_candidate_count >= 1

    def test_consistency_config_present(self):
        assert 0.0 <= matching_config.consistency_penalty_rate <= 0.5
        assert "low" in matching_config.consistency_severity_weights
        assert "medium" in matching_config.consistency_severity_weights
        assert "high" in matching_config.consistency_severity_weights

    def test_confidence_config_present(self):
        assert 0.0 <= matching_config.min_confidence_threshold <= 1.0

    def test_cache_ttl_config_present(self):
        assert matching_config.dimension_cache_ttl > 0
        assert matching_config.position_cache_ttl > 0
        assert matching_config.description_cache_ttl > 0

    def test_llm_token_limits_sensible(self):
        assert matching_config.llm_max_tokens_dimension_discovery >= 500
        assert matching_config.llm_max_tokens_question_mapping >= 500
        assert matching_config.llm_max_tokens_voter_value_description >= 100
        assert matching_config.llm_max_tokens_position_description >= 100

    def test_intensity_multipliers_present(self):
        multipliers = matching_config.intensity_multipliers
        assert "strongly agree" in multipliers
        assert "agree" in multipliers
        assert "neutral" in multipliers
        assert "disagree" in multipliers
        assert "strongly disagree" in multipliers
        # Strong opinions should have higher multipliers than mild ones
        assert multipliers["strongly agree"] > multipliers["agree"]
        assert multipliers["strongly disagree"] > multipliers["disagree"]


# ── Position Inference Tests ──────────────────────────────────────────────────

class TestPositionInferenceService:
    """Tests for the position inference service."""

    def test_is_structured_answer_boolean(self, position_inference_service):
        """Booleans are always structured answers."""
        assert position_inference_service._is_structured_answer(True) is True
        assert position_inference_service._is_structured_answer(False) is True

    def test_is_structured_answer_yes_no(self, position_inference_service):
        """yes/no text answers are structured."""
        assert position_inference_service._is_structured_answer("yes") is True
        assert position_inference_service._is_structured_answer("no") is True
        assert position_inference_service._is_structured_answer("Yes") is True
        assert position_inference_service._is_structured_answer("NO") is True

    def test_is_structured_answer_agree_disagree(self, position_inference_service):
        """agree/disagree variants are structured."""
        assert position_inference_service._is_structured_answer("agree") is True
        assert position_inference_service._is_structured_answer("strongly agree") is True
        assert position_inference_service._is_structured_answer("disagree") is True
        assert position_inference_service._is_structured_answer("strongly disagree") is True

    def test_is_structured_answer_list(self, position_inference_service):
        """Multi-select list answers are structured (need question context)."""
        assert position_inference_service._is_structured_answer(["Option A", "Option B"]) is True

    def test_is_structured_answer_dict(self, position_inference_service):
        """Dict answers are structured (need question context)."""
        assert position_inference_service._is_structured_answer({"priority": "high"}) is True

    def test_is_structured_answer_free_text(self, position_inference_service):
        """Long free-text answers are NOT structured."""
        assert position_inference_service._is_structured_answer(
            "I strongly believe we need more investment in public education"
        ) is False

    def test_is_policy_relevant_boolean(self, position_inference_service):
        """Booleans are always policy-relevant."""
        assert position_inference_service._is_policy_relevant(True) is True
        assert position_inference_service._is_policy_relevant(False) is True

    def test_is_policy_relevant_long_text(self, position_inference_service):
        """Long enough text is policy-relevant."""
        assert position_inference_service._is_policy_relevant(
            "I support increased funding for public schools"
        ) is True

    def test_is_policy_relevant_numeric_string(self, position_inference_service):
        """Pure numeric strings (e.g. age) are NOT policy-relevant."""
        assert position_inference_service._is_policy_relevant("32") is False
        assert position_inference_service._is_policy_relevant("100") is False

    def test_extract_basic_position_boolean_true(self, position_inference_service, education_dimension):
        """True boolean → high position score."""
        result = position_inference_service._extract_basic_position(True, education_dimension)
        assert result["score"] == 80.0
        assert result["method"] == "boolean"

    def test_extract_basic_position_boolean_false(self, position_inference_service, education_dimension):
        """False boolean → low position score."""
        result = position_inference_service._extract_basic_position(False, education_dimension)
        assert result["score"] == 20.0
        assert result["method"] == "boolean"

    def test_extract_basic_position_strongly_agree(self, position_inference_service, education_dimension):
        """Strongly agree → score 90."""
        result = position_inference_service._extract_basic_position("Strongly Agree", education_dimension)
        assert result["score"] == 90.0
        assert result["method"] == "strong_positive"

    def test_extract_basic_position_strongly_disagree(self, position_inference_service, education_dimension):
        """Strongly disagree → score 10."""
        result = position_inference_service._extract_basic_position("Strongly Disagree", education_dimension)
        assert result["score"] == 10.0
        assert result["method"] == "strong_negative"

    def test_extract_basic_position_neutral(self, position_inference_service, education_dimension):
        """Neutral answers → score 50."""
        result = position_inference_service._extract_basic_position("Neutral", education_dimension)
        assert result["score"] == 50.0
        assert result["method"] == "neutral"

    def test_extract_basic_position_list(self, position_inference_service, education_dimension):
        """List answers → structured_needs_context (goes to LLM path)."""
        result = position_inference_service._extract_basic_position(
            ["STEM", "Arts", "Special education"], education_dimension
        )
        assert result["method"] == "structured_needs_context"

    def test_extract_basic_position_irrelevant_returns_sentinel(
        self, position_inference_service, education_dimension
    ):
        """Very short non-structured text → irrelevant sentinel."""
        result = position_inference_service._extract_basic_position("Hi", education_dimension)
        assert result["method"] == "irrelevant"
        assert result["score"] == -1.0

    @pytest.mark.asyncio
    @patch('app.services.position_inference_service.llm_service')
    @patch('app.services.position_inference_service.cache_service')
    async def test_infer_policy_position_returns_none_for_irrelevant_answer(
        self, mock_cache, mock_llm, position_inference_service, education_dimension
    ):
        """Answers with no policy content should return None."""
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock(return_value=True)

        result = await position_inference_service.infer_policy_position(
            question="What is your name?",
            answer="Jo",  # Too short, not a structured token
            comment="",
            dimension=education_dimension,
            person_type="voter",
        )
        assert result is None

    @pytest.mark.asyncio
    @patch('app.services.position_inference_service.llm_service')
    @patch('app.services.position_inference_service.cache_service')
    async def test_infer_policy_position_returns_position_for_boolean(
        self, mock_cache, mock_llm, position_inference_service, education_dimension
    ):
        """Boolean answers should return a valid PolicyPosition via LLM context path."""
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock(return_value=True)
        mock_llm.call_llm = AsyncMock(return_value='{"position_score": 82.0, "confidence": 0.88, "reasoning": "Yes to supporting education funding means high support."}')
        mock_llm._extract_json_from_response = lambda x: {
            "position_score": 82.0,
            "confidence": 0.88,
            "reasoning": "Yes to supporting education funding means high support.",
        }

        result = await position_inference_service.infer_policy_position(
            question="Do you support increased funding for public schools?",
            answer=True,
            comment="",
            dimension=education_dimension,
            person_type="voter",
        )
        assert result is not None
        assert isinstance(result, PolicyPosition)
        assert 0 <= result.position_score <= 100
        assert result.dimension_id == "education_funding"


# ── Consistency Analyzer Tests ────────────────────────────────────────────────

class TestConsistencyAnalyzerService:
    """Tests for the V2 generalized consistency analyzer."""

    def test_no_tensions_returns_score_one(self, consistency_analyzer):
        """Fully consistent positions should return score of 1.0."""
        positions = [
            PolicyPosition(
                dimension_id="education_funding",
                position_score=80.0,
                confidence=0.9,
                intensity_multiplier=1.5,
                reasoning="Strong support for education",
                source_question="Q1",
                source_answer="Strongly Agree",
            ),
            PolicyPosition(
                dimension_id="tax_revenue",
                position_score=75.0,
                confidence=0.9,
                intensity_multiplier=1.0,
                reasoning="Supports revenue for programs",
                source_question="Q2",
                source_answer="Agree",
            ),
        ]
        tensions, score = consistency_analyzer.analyze_position_consistency(positions)
        # High spending + high revenue = no conflict
        assert score == 1.0 or len(tensions) == 0

    def test_spend_revenue_conflict_detected(self, consistency_analyzer):
        """High spending desire + low revenue support = HIGH tension."""
        positions = [
            PolicyPosition(
                dimension_id="education_funding",  # spending keyword: "fund"
                position_score=85.0,
                confidence=0.9,
                intensity_multiplier=2.0,
                reasoning="Wants much more education funding",
                source_question="Q1",
                source_answer="Strongly Agree",
            ),
            PolicyPosition(
                dimension_id="tax_revenue",  # revenue keyword: "tax", "revenue"
                position_score=20.0,
                confidence=0.9,
                intensity_multiplier=2.0,
                reasoning="Strongly opposes tax increases",
                source_question="Q2",
                source_answer="Strongly Disagree",
            ),
        ]
        tensions, score = consistency_analyzer.analyze_position_consistency(positions)
        assert len(tensions) > 0
        assert score < 1.0
        assert tensions[0].severity == TensionSeverity.HIGH

    def test_central_local_conflict_detected(self, consistency_analyzer):
        """Strong central control + strong local autonomy = MEDIUM tension."""
        positions = [
            PolicyPosition(
                dimension_id="government_mandate",  # central keyword: "government", "mandate"
                position_score=80.0,
                confidence=0.9,
                intensity_multiplier=1.5,
                reasoning="Supports government mandates",
                source_question="Q1",
                source_answer="Agree",
            ),
            PolicyPosition(
                dimension_id="local_autonomy",  # local keyword: "local", "autonomy"
                position_score=80.0,
                confidence=0.9,
                intensity_multiplier=1.5,
                reasoning="Supports local control",
                source_question="Q2",
                source_answer="Agree",
            ),
        ]
        tensions, score = consistency_analyzer.analyze_position_consistency(positions)
        assert len(tensions) > 0
        assert score < 1.0

    def test_disabled_consistency_returns_no_tensions(self, consistency_analyzer):
        """When consistency analysis is disabled, no tensions should be returned."""
        positions = [
            PolicyPosition(
                dimension_id="education_funding",
                position_score=85.0,
                confidence=0.9,
                intensity_multiplier=2.0,
                reasoning="Wants much more education funding",
                source_question="Q1",
                source_answer="Strongly Agree",
            ),
            PolicyPosition(
                dimension_id="tax_revenue",
                position_score=20.0,
                confidence=0.9,
                intensity_multiplier=2.0,
                reasoning="Strongly opposes tax increases",
                source_question="Q2",
                source_answer="Strongly Disagree",
            ),
        ]
        with patch.object(
            type(matching_config), 'enable_consistency_analysis',
            new_callable=lambda: property(lambda self: False)
        ):
            tensions, score = consistency_analyzer.analyze_position_consistency(positions)
            assert tensions == []
            assert score == 1.0

    def test_apply_consistency_penalty(self, consistency_analyzer):
        """Consistency penalty should reduce the match score proportionally."""
        base_score = 80.0
        consistency_score = 0.5  # 50% consistent

        adjusted = consistency_analyzer.apply_consistency_penalty(base_score, consistency_score)
        assert adjusted < base_score
        assert adjusted > 0

    def test_fully_consistent_no_penalty(self, consistency_analyzer):
        """A fully consistent profile (score=1.0) should not be penalised."""
        base_score = 80.0
        adjusted = consistency_analyzer.apply_consistency_penalty(base_score, 1.0)
        assert adjusted == base_score


# ── Enhanced Matching Calculator Tests ───────────────────────────────────────

class TestEnhancedMatchingCalculatorService:
    """Tests for the enhanced matching calculator."""

    def test_calculate_base_match_percentage_perfect_alignment(self, calculator):
        """Perfect dimension alignment should yield 100%."""
        from app.schemas.policy_matching_schema import DimensionMatch
        dimension_matches = [
            DimensionMatch(
                dimension_id="education_funding",
                dimension_name="Education Funding",
                voter_position=80.0,
                candidate_position=80.0,
                alignment_score=1.0,
                alignment_level="Strongly Aligned",
                voter_position_description="Strong support",
                candidate_position_description="Strong support",
                weight=1.0,
            )
        ]
        result = calculator._calculate_base_match_percentage(dimension_matches)
        assert result == 100.0

    def test_calculate_base_match_percentage_opposite_positions(self, calculator):
        """Opposite positions should yield a low match percentage."""
        from app.schemas.policy_matching_schema import DimensionMatch
        dimension_matches = [
            DimensionMatch(
                dimension_id="education_funding",
                dimension_name="Education Funding",
                voter_position=90.0,
                candidate_position=10.0,
                alignment_score=0.2,
                alignment_level="Weakly Aligned",
                voter_position_description="Strong support",
                candidate_position_description="Strong opposition",
                weight=1.0,
            )
        ]
        result = calculator._calculate_base_match_percentage(dimension_matches)
        assert result < 50.0

    def test_confidence_filter_excludes_low_confidence_positions(self, calculator):
        """Positions below min_confidence_threshold should be excluded."""
        from app.schemas.policy_matching_schema import PersonPolicyProfile

        voter_profile = PersonPolicyProfile(
            person_id="v001",
            person_type="voter",
            election_id="e001",
            policy_positions=[
                PolicyPosition(
                    dimension_id="education_funding",
                    position_score=80.0,
                    confidence=0.05,  # Below threshold of 0.3
                    intensity_multiplier=1.0,
                    reasoning="Low confidence position",
                    source_question="Q1",
                    source_answer="maybe",
                )
            ],
        )
        candidate_profile = PersonPolicyProfile(
            person_id="c001",
            person_type="candidate",
            election_id="e001",
            policy_positions=[
                PolicyPosition(
                    dimension_id="education_funding",
                    position_score=80.0,
                    confidence=0.9,
                    intensity_multiplier=2.0,
                    reasoning="Strong position",
                    source_question="Q1",
                    source_answer="Strongly Agree",
                )
            ],
        )

        # The voter's low-confidence position should be filtered out,
        # leaving 0 overlapping dimensions → should return 0% match
        import asyncio

        async def run():
            return await calculator.calculate_enhanced_match(voter_profile, candidate_profile)

        # We just verify it doesn't raise — actual score depends on config
        # The key is the low-confidence position doesn't cause an error
        with patch('app.services.enhanced_matching_calculator_service.cache_service') as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock(return_value=True)
            with patch('app.services.enhanced_matching_calculator_service.llm_service') as mock_llm:
                mock_llm.call_llm = AsyncMock(return_value='{}')
                mock_llm._extract_json_from_response = lambda x: {}
                result = asyncio.get_event_loop().run_until_complete(run())
                # With only 0 overlapping dimensions above threshold, match should be 0
                assert result.overall_match_percentage == 0


# ── Policy Dimension Discovery Tests ─────────────────────────────────────────

class TestPolicyDimensionDiscoveryService:
    """Tests for the V2 generic fallback dimension discovery."""

    @pytest.mark.asyncio
    async def test_fallback_dimensions_are_generic(self, discovery_service):
        """Fallback dimensions should be derived from question content, not hardcoded."""
        questions = [
            "Should the government increase funding for public healthcare?",
            "Do you support higher taxes on corporations?",
            "Should local communities have more control over zoning decisions?",
            "Do you support expanding access to affordable housing?",
            "Should the government mandate renewable energy standards?",
        ]
        dimensions = await discovery_service._create_fallback_dimensions(questions)

        assert len(dimensions) > 0
        assert len(dimensions) <= matching_config.max_policy_dimensions

        # None of the fallback dimension IDs should be education-specific
        education_specific_ids = {"student_support", "school_safety"}
        for dim in dimensions:
            assert dim.dimension_id not in education_specific_ids

    @pytest.mark.asyncio
    async def test_fallback_dimensions_reflect_question_content(self, discovery_service):
        """Fallback dimensions should be ranked by relevance to the questions."""
        # Questions heavily about taxation and revenue
        questions = [
            "Do you support raising the corporate tax rate?",
            "Should capital gains taxes be increased?",
            "Do you support a wealth tax on billionaires?",
            "Should tax loopholes for large corporations be closed?",
            "Do you support higher income tax for top earners?",
        ]
        dimensions = await discovery_service._create_fallback_dimensions(questions)

        # The spending/funding or revenue dimension should rank high
        dimension_ids = [d.dimension_id for d in dimensions]
        # At least one revenue or spending related dimension should appear
        revenue_related = {"spending_and_funding", "public_services"}
        assert any(did in revenue_related for did in dimension_ids)

    @pytest.mark.asyncio
    async def test_fallback_always_returns_at_least_one_dimension(self, discovery_service):
        """Even with no recognisable keywords, we should get at least one dimension."""
        questions = ["???", "asdkjhaskjdh", "xyzzy"]
        dimensions = await discovery_service._create_fallback_dimensions(questions)
        assert len(dimensions) >= 1
