# tests/test_matching_engine_service.py
#
# V2 test suite for the matching engine service.
# Tests focus on the V2 pipeline:
#   - Voter quality gate
#   - Context-aware answer interpretation
#   - Policy profile building
#   - Match categorisation

import pytest
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock

from app.schemas.voters_schema import (
    VoterSubmissionSchema,
    VoterResponseItemSchema,
    CandidateMatchSchema,
    MatchResultsResponseSchema,
    VoterValueProfileSchema,
)
from app.schemas.candidate_schema import (
    CandidateResponseSchema,
    CandidateResponseItemSchema,
)
from app.schemas.policy_matching_schema import (
    PersonPolicyProfile,
    PolicyPosition,
    ElectionPolicyAnalysis,
    PolicyDimension,
    QuestionDimensionMapping,
)
from app.services.matching_engine_service import MatchingEngineService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def matching_engine():
    """Create a MatchingEngineService instance for testing."""
    return MatchingEngineService()


@pytest.fixture
def voter_submission():
    """A voter submission with enough policy-relevant responses to pass the quality gate."""
    return VoterSubmissionSchema(
        election_id="e001",
        citizen_id="v001",
        responses=[
            VoterResponseItemSchema(
                question_id="q001",
                question="Should students have access to a language immersion middle school?",
                answer="Strongly Agree",
            ),
            VoterResponseItemSchema(
                question_id="q002",
                question="Which educational programs should receive increased funding?",
                answer=["STEM initiatives", "Special education", "Arts and music"],
            ),
            VoterResponseItemSchema(
                question_id="q003",
                question="Do you think the council member should prioritize mental health resources for students?",
                answer="Strongly Agree",
            ),
            VoterResponseItemSchema(
                question_id="q004",
                question="Do you believe School Resource Officers effectively keep schools safe?",
                answer="Strongly Disagree",
            ),
            VoterResponseItemSchema(
                question_id="q005",
                question="Should the council member actively pass legislation benefiting students and families?",
                answer="Agree",
            ),
            VoterResponseItemSchema(
                question_id="q006",
                question="Has education in your neighborhood improved over the last 20 years?",
                answer="Disagree",
            ),
            VoterResponseItemSchema(
                question_id="q007",
                question="What is your top priority for public schools?",
                answer="Increase funding for public schools",
            ),
            VoterResponseItemSchema(
                question_id="q008",
                question="Do you support vocational training in schools?",
                answer=True,
            ),
        ],
        completed_at=datetime.now(),
    )


@pytest.fixture
def insufficient_voter_submission():
    """A voter submission that should FAIL the quality gate (only personal info)."""
    return VoterSubmissionSchema(
        election_id="e001",
        citizen_id="v002",
        responses=[
            VoterResponseItemSchema(
                question_id="q001",
                question="What is your name?",
                answer="John Smith",
            ),
            VoterResponseItemSchema(
                question_id="q002",
                question="What is your age?",
                answer="32",
            ),
        ],
        completed_at=datetime.now(),
    )


@pytest.fixture
def eligible_candidate():
    """A candidate eligible for matching (completed profile and questionnaire)."""
    return CandidateResponseSchema(
        candidate_id="c001",
        election_id="e001",
        has_completed_profile=True,
        has_completed_questionnaire=True,
        responses=[
            CandidateResponseItemSchema(
                id="r001",
                question="Should students have access to a language immersion middle school?",
                answer="Strongly Agree",
                comment="Language immersion programs are crucial for our students' future success.",
                election_id="e001",
            ),
            CandidateResponseItemSchema(
                id="r002",
                question="Which educational programs should receive increased funding?",
                answer=["STEM initiatives", "Arts and music", "Special education"],
                comment="We need balanced funding across multiple educational areas.",
                election_id="e001",
            ),
            CandidateResponseItemSchema(
                id="r003",
                question="Do you think the council member should prioritize mental health resources for students?",
                answer="Strongly Agree",
                comment="Student mental health must be a top priority.",
                election_id="e001",
            ),
            CandidateResponseItemSchema(
                id="r004",
                question="Do you believe School Resource Officers effectively keep schools safe?",
                answer="Disagree",
                comment="We need more community-based approaches to school safety.",
                election_id="e001",
            ),
            CandidateResponseItemSchema(
                id="r005",
                question="Should the council member actively pass legislation benefiting students and families?",
                answer="Strongly Agree",
                comment="Proactive legislation is essential for improving education.",
                election_id="e001",
            ),
        ],
    )


@pytest.fixture
def ineligible_candidate():
    """A candidate that is NOT eligible for matching (incomplete profile)."""
    return CandidateResponseSchema(
        candidate_id="c002",
        election_id="e001",
        has_completed_profile=False,
        has_completed_questionnaire=False,
        responses=[],
    )


# ── Quality Gate Tests ────────────────────────────────────────────────────────

class TestVoterQualityGate:
    """Tests for the voter response quality gate."""

    def test_sufficient_responses_pass(self, matching_engine, voter_submission):
        """A submission with enough policy responses should pass."""
        result = matching_engine._check_voter_response_quality(voter_submission)
        assert result == ""  # Empty string = passed

    def test_personal_info_only_fails(self, matching_engine, insufficient_voter_submission):
        """A submission with only name/age should fail the quality gate."""
        result = matching_engine._check_voter_response_quality(insufficient_voter_submission)
        assert result != ""  # Non-empty = failed
        assert "policy-relevant" in result.lower() or "required" in result.lower()

    def test_boolean_answers_count(self, matching_engine):
        """Boolean answers on non-personal questions should count."""
        submission = VoterSubmissionSchema(
            election_id="e001",
            citizen_id="v003",
            responses=[
                VoterResponseItemSchema(
                    question_id="q001",
                    question="Do you support increased school funding?",
                    answer=True,
                ),
                VoterResponseItemSchema(
                    question_id="q002",
                    question="Should vocational training be offered in schools?",
                    answer=False,
                ),
                VoterResponseItemSchema(
                    question_id="q003",
                    question="Do you support mental health programs in schools?",
                    answer=True,
                ),
            ],
        )
        result = matching_engine._check_voter_response_quality(submission)
        assert result == ""

    def test_structured_text_answers_count(self, matching_engine):
        """Short structured answers like 'yes', 'agree' should count."""
        submission = VoterSubmissionSchema(
            election_id="e001",
            citizen_id="v004",
            responses=[
                VoterResponseItemSchema(
                    question_id="q001",
                    question="Do you support increased funding for schools?",
                    answer="yes",
                ),
                VoterResponseItemSchema(
                    question_id="q002",
                    question="Should the council prioritize mental health?",
                    answer="strongly agree",
                ),
                VoterResponseItemSchema(
                    question_id="q003",
                    question="Do you support vocational training?",
                    answer="agree",
                ),
            ],
        )
        result = matching_engine._check_voter_response_quality(submission)
        assert result == ""

    def test_pure_numeric_answer_does_not_count(self, matching_engine):
        """A pure numeric text answer (e.g. age) should not count as policy-relevant."""
        submission = VoterSubmissionSchema(
            election_id="e001",
            citizen_id="v005",
            responses=[
                VoterResponseItemSchema(
                    question_id="q001",
                    question="What is your budget preference?",
                    answer="32",  # Looks numeric — should not count
                ),
                VoterResponseItemSchema(
                    question_id="q002",
                    question="What is your name?",
                    answer="John",  # Personal info — should not count
                ),
            ],
        )
        result = matching_engine._check_voter_response_quality(submission)
        assert result != ""


# ── Process Voter Submission Tests ────────────────────────────────────────────

class TestProcessVoterSubmission:
    """Integration-style tests for the full matching pipeline."""

    @pytest.mark.asyncio
    @patch('app.services.matching_engine_service.candidate_service')
    @patch('app.services.matching_engine_service.policy_dimension_discovery_service')
    @patch('app.services.matching_engine_service.enhanced_matching_calculator_service')
    @patch('app.services.matching_engine_service.position_inference_service')
    async def test_returns_match_results_schema(
        self,
        mock_position_svc,
        mock_calculator_svc,
        mock_discovery_svc,
        mock_candidate_svc,
        matching_engine,
        voter_submission,
        eligible_candidate,
    ):
        """A valid submission should return a MatchResultsResponseSchema."""
        # Mock candidate service
        mock_candidate_svc.get_candidates_for_election = AsyncMock(
            return_value=[eligible_candidate]
        )

        # Mock dimension discovery
        mock_dimension = PolicyDimension(
            dimension_id="education_policy",
            name="Education Policy",
            description="Education funding and access",
            policy_spectrum_description="Low Support (0) to High Support (100)",
            keywords=["education", "school", "funding"],
            confidence=0.9,
        )
        mock_mapping = QuestionDimensionMapping(
            question="Should students have access to a language immersion middle school?",
            primary_dimension_id="education_policy",
            secondary_dimension_ids=[],
            primary_weight=1.0,
            secondary_weights={},
            mapping_confidence=0.9,
        )
        mock_analysis = ElectionPolicyAnalysis(
            election_id="e001",
            discovered_dimensions=[mock_dimension],
            question_mappings=[mock_mapping],
            discovery_confidence=0.9,
            analysis_timestamp=str(datetime.now()),
        )
        mock_discovery_svc.discover_election_policy_dimensions = AsyncMock(
            return_value=mock_analysis
        )

        # Mock position inference
        mock_position = PolicyPosition(
            dimension_id="education_policy",
            position_score=80.0,
            confidence=0.85,
            intensity_multiplier=2.0,
            reasoning="Strongly supports education access",
            source_question="Should students have access to a language immersion middle school?",
            source_answer="Strongly Agree",
        )
        mock_position_svc.infer_policy_position = AsyncMock(return_value=mock_position)

        # Mock calculator
        mock_match = CandidateMatchSchema(
            candidate_id="c001",
            match_percentage=82,
            match_strength_visual=0.82,
            match_category="TOP",
            top_aligned_issues=["Education Policy"],
            issue_matches=[],
            overall_explanation="Strong alignment on education policy.",
        )
        mock_calculator_svc.calculate_enhanced_match = AsyncMock(
            return_value=MagicMock(
                voter_id="v001",
                candidate_id="c001",
                overall_match_percentage=82,
                confidence_weighted_percentage=78,
                dimension_matches=[],
                consistency_penalty_applied=0.0,
                match_explanation="Strong alignment on education policy.",
                top_aligned_dimensions=["Education Policy"],
            )
        )

        result = await matching_engine.process_voter_submission(voter_submission)

        assert isinstance(result, MatchResultsResponseSchema)
        assert result.citizen_id == voter_submission.citizen_id
        assert result.election_id == voter_submission.election_id
        assert result.processing_method != "error"

    @pytest.mark.asyncio
    @patch('app.services.matching_engine_service.candidate_service')
    async def test_insufficient_responses_returns_empty_matches(
        self, mock_candidate_svc, matching_engine, insufficient_voter_submission
    ):
        """A submission that fails the quality gate should return empty matches."""
        mock_candidate_svc.get_candidates_for_election = AsyncMock(return_value=[])

        result = await matching_engine.process_voter_submission(insufficient_voter_submission)

        assert isinstance(result, MatchResultsResponseSchema)
        assert result.citizen_id == insufficient_voter_submission.citizen_id
        assert len(result.matches) == 0
        assert result.processing_method == "insufficient_responses"
        assert result.confidence_score == 0.0

    @pytest.mark.asyncio
    @patch('app.services.matching_engine_service.candidate_service')
    async def test_no_candidates_returns_empty_matches(
        self, mock_candidate_svc, matching_engine, voter_submission
    ):
        """When no candidates are found, matches should be empty."""
        mock_candidate_svc.get_candidates_for_election = AsyncMock(return_value=[])

        result = await matching_engine.process_voter_submission(voter_submission)

        assert isinstance(result, MatchResultsResponseSchema)
        assert len(result.matches) == 0
        assert result.processing_method == "no_candidates"

    @pytest.mark.asyncio
    @patch('app.services.matching_engine_service.candidate_service')
    async def test_ineligible_candidate_gets_zero_match(
        self, mock_candidate_svc, matching_engine, voter_submission, ineligible_candidate
    ):
        """Ineligible candidates should receive 0% match and UNMATCH category."""
        mock_candidate_svc.get_candidates_for_election = AsyncMock(
            return_value=[ineligible_candidate]
        )

        result = await matching_engine.process_voter_submission(voter_submission)

        assert isinstance(result, MatchResultsResponseSchema)
        assert len(result.matches) == 1
        assert result.matches[0].match_percentage == 0
        assert result.matches[0].match_category == "UNMATCH"
        assert result.matches[0].candidate_id == ineligible_candidate.candidate_id


# ── Match Categorisation Tests ────────────────────────────────────────────────

class TestMatchCategorisation:
    """Tests for the _assign_match_categories method."""

    def test_top_candidates_assigned_correctly(self, matching_engine):
        """Top N candidates by score should get TOP category."""
        matches = [
            CandidateMatchSchema(
                candidate_id=f"c00{i}",
                match_percentage=90 - (i * 10),
                match_strength_visual=(90 - (i * 10)) / 100,
                match_category="PENDING",
                top_aligned_issues=[],
                issue_matches=[],
            )
            for i in range(5)
        ]

        matching_engine._assign_match_categories(matches)

        top_matches = [m for m in matches if m.match_category == "TOP"]
        other_matches = [m for m in matches if m.match_category == "OTHER"]

        assert len(top_matches) == 3  # top_candidate_count default
        assert len(other_matches) == 2

    def test_zero_percent_candidates_get_unmatch(self, matching_engine):
        """Candidates with 0% match should be UNMATCH."""
        matches = [
            CandidateMatchSchema(
                candidate_id="c001",
                match_percentage=75,
                match_strength_visual=0.75,
                match_category="PENDING",
                top_aligned_issues=[],
                issue_matches=[],
            ),
            CandidateMatchSchema(
                candidate_id="c002",
                match_percentage=0,
                match_strength_visual=0.0,
                match_category="UNMATCH",
                top_aligned_issues=[],
                issue_matches=[],
            ),
        ]

        matching_engine._assign_match_categories(matches)

        assert matches[0].match_category == "TOP"
        assert matches[1].match_category == "UNMATCH"


# ── Alignment Level Tests ─────────────────────────────────────────────────────

class TestAlignmentLevel:
    """Tests for the _get_alignment_level utility method."""

    def test_strongly_aligned(self, matching_engine):
        assert matching_engine._get_alignment_level(0.9) == "Strongly Aligned"
        assert matching_engine._get_alignment_level(0.8) == "Strongly Aligned"

    def test_moderately_aligned(self, matching_engine):
        assert matching_engine._get_alignment_level(0.7) == "Moderately Aligned"
        assert matching_engine._get_alignment_level(0.6) == "Moderately Aligned"

    def test_somewhat_aligned(self, matching_engine):
        assert matching_engine._get_alignment_level(0.5) == "Somewhat Aligned"
        assert matching_engine._get_alignment_level(0.4) == "Somewhat Aligned"

    def test_weakly_aligned(self, matching_engine):
        assert matching_engine._get_alignment_level(0.3) == "Weakly Aligned"
        assert matching_engine._get_alignment_level(0.0) == "Weakly Aligned"


# ── Extract Questions Tests ───────────────────────────────────────────────────

class TestExtractAllQuestions:
    """Tests for the _extract_all_questions static method."""

    def test_combines_voter_and_candidate_questions(
        self, matching_engine, voter_submission, eligible_candidate
    ):
        """Should return unique questions from both voter and candidates."""
        questions = matching_engine._extract_all_questions(
            voter_submission, [eligible_candidate]
        )
        assert isinstance(questions, list)
        assert len(questions) > 0
        # All voter questions should be present
        for response in voter_submission.responses:
            assert response.question in questions

    def test_deduplicates_questions(self, matching_engine, voter_submission, eligible_candidate):
        """Common questions should not be duplicated."""
        questions = matching_engine._extract_all_questions(
            voter_submission, [eligible_candidate]
        )
        assert len(questions) == len(set(questions))


# ── Zero Match Result Tests ───────────────────────────────────────────────────

class TestCreateZeroMatchResult:
    """Tests for the _create_zero_match_result method."""

    def test_zero_match_has_correct_fields(self, matching_engine, ineligible_candidate):
        """Zero match result should have 0% and UNMATCH category."""
        result = matching_engine._create_zero_match_result(ineligible_candidate)

        assert result.candidate_id == ineligible_candidate.candidate_id
        assert result.match_percentage == 0
        assert result.match_strength_visual == 0.0
        assert result.match_category == "UNMATCH"
        assert result.top_aligned_issues == []
        assert result.issue_matches == []
        assert result.overall_explanation is not None
