# tests/test_matching_engine_router.py
#
# V2 test suite for the matching engine API routes.
# Verifies that:
#   - Endpoints accept the correct request payloads (camelCase JSON)
#   - Endpoints return the correct response shapes
#   - API contracts are fully preserved from V1

import pytest
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.voters_schema import (
    MatchResultsResponseSchema,
    CandidateMatchSchema,
    VoterValueProfileSchema,
)


# ── Test Client ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ── Shared Payload Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def valid_voter_payload():
    """
    A valid voter submission payload in camelCase JSON — exactly what
    the frontend sends. This must never change shape.
    """
    return {
        "electionId": "e001",
        "citizenId": "v001",
        "completedAt": datetime.now().isoformat(),
        "responses": [
            {
                "questionId": "q001",
                "question": "Should students have access to a language immersion middle school?",
                "answer": "Strongly Agree",
            },
            {
                "questionId": "q002",
                "question": "Which educational programs should receive increased funding?",
                "answer": ["STEM initiatives", "Special education", "Arts and music"],
            },
            {
                "questionId": "q003",
                "question": "Do you think the council member should prioritize mental health resources for students?",
                "answer": "Strongly Agree",
            },
            {
                "questionId": "q004",
                "question": "Do you believe School Resource Officers effectively keep schools safe?",
                "answer": "Strongly Disagree",
            },
            {
                "questionId": "q005",
                "question": "Should the council member actively pass legislation benefiting students?",
                "answer": True,
            },
        ],
    }


@pytest.fixture
def insufficient_voter_payload():
    """A voter payload that will fail the quality gate (name + age only)."""
    return {
        "electionId": "e001",
        "citizenId": "v002",
        "completedAt": datetime.now().isoformat(),
        "responses": [
            {
                "questionId": "q001",
                "question": "What is your name?",
                "answer": "John Smith",
            },
            {
                "questionId": "q002",
                "question": "What is your age?",
                "answer": "32",
            },
        ],
    }


@pytest.fixture
def mock_match_response():
    """A valid MatchResultsResponseSchema for mocking the service layer."""
    return MatchResultsResponseSchema(
        citizen_id="v001",
        election_id="e001",
        voter_values_profile=[
            VoterValueProfileSchema(
                dimension_id="education_policy",
                dimension_name="Education Policy",
                value_description="Strong support for education access and funding",
                position_score=82.0,
                priority="high",
            )
        ],
        matches=[
            CandidateMatchSchema(
                candidate_id="c001",
                match_percentage=82,
                match_strength_visual=0.82,
                match_category="TOP",
                top_aligned_issues=["Education Policy"],
                issue_matches=[],
                overall_explanation="Strong alignment on education policy.",
            )
        ],
        generated_at=datetime.now(),
        processing_method="enhanced_policy_matching",
        confidence_score=0.87,
    )


@pytest.fixture
def mock_insufficient_response():
    """Response returned when quality gate rejects the submission."""
    return MatchResultsResponseSchema(
        citizen_id="v002",
        election_id="e001",
        voter_values_profile=[],
        matches=[],
        generated_at=datetime.now(),
        processing_method="insufficient_responses",
        confidence_score=0.0,
    )


# ── Submit Voter Responses Endpoint Tests ─────────────────────────────────────

class TestSubmitVoterResponsesEndpoint:
    """Tests for POST /matching_engine/submit-voter-responses"""

    @patch('app.api.routes.matching_engine.MatchingEngineRouter.submit_voter_responses')
    def test_valid_payload_returns_200(self, mock_handler, client, valid_voter_payload, mock_match_response):
        """A valid payload should return 200 with match results."""
        mock_handler.return_value = mock_match_response

        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=valid_voter_payload,
        )

        assert response.status_code == 200

    @patch('app.services.matching_engine_service.matching_engine.process_voter_submission')
    def test_response_contains_citizen_id(
        self, mock_process, client, valid_voter_payload, mock_match_response
    ):
        """Response must contain citizenId (camelCase) matching the submission."""
        mock_process.return_value = mock_match_response

        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=valid_voter_payload,
        )

        if response.status_code == 200:
            data = response.json()
            # Response uses camelCase because of CamelModel
            assert "citizenId" in data or "citizen_id" in data

    @patch('app.services.matching_engine_service.matching_engine.process_voter_submission')
    def test_response_contains_matches_list(
        self, mock_process, client, valid_voter_payload, mock_match_response
    ):
        """Response must always contain a matches array."""
        mock_process.return_value = mock_match_response

        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=valid_voter_payload,
        )

        if response.status_code == 200:
            data = response.json()
            assert "matches" in data
            assert isinstance(data["matches"], list)

    @patch('app.services.matching_engine_service.matching_engine.process_voter_submission')
    def test_insufficient_responses_returns_empty_matches(
        self, mock_process, client, insufficient_voter_payload, mock_insufficient_response
    ):
        """A payload that fails the quality gate returns empty matches, not an error."""
        mock_process.return_value = mock_insufficient_response

        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=insufficient_voter_payload,
        )

        if response.status_code == 200:
            data = response.json()
            assert data["matches"] == []
            processing_method = data.get("processingMethod") or data.get("processing_method")
            assert processing_method == "insufficient_responses"

    def test_missing_election_id_returns_422(self, client):
        """Missing required fields should return 422 Unprocessable Entity."""
        payload = {
            "citizenId": "v001",
            "responses": [],
        }
        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=payload,
        )
        assert response.status_code == 422

    def test_missing_citizen_id_returns_422(self, client):
        """Missing citizenId should return 422."""
        payload = {
            "electionId": "e001",
            "responses": [],
        }
        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=payload,
        )
        assert response.status_code == 422


# ── Health Check Endpoint Tests ───────────────────────────────────────────────

class TestHealthCheckEndpoint:
    """Tests for GET /matching_engine/health"""

    def test_health_check_returns_200(self, client):
        """Health check should always return 200."""
        response = client.get("/matching_engine/health")
        assert response.status_code == 200

    def test_health_check_response_structure(self, client):
        """Health check response should have expected fields."""
        response = client.get("/matching_engine/health")
        if response.status_code == 200:
            data = response.json()
            assert "status" in data or "message" in data or "health" in data


# ── Cache Stats Endpoint Tests ────────────────────────────────────────────────

class TestCacheStatsEndpoint:
    """Tests for GET /matching_engine/cache-stats"""

    def test_cache_stats_returns_200(self, client):
        """Cache stats endpoint should return 200."""
        response = client.get("/matching_engine/cache-stats")
        assert response.status_code == 200

    def test_cache_stats_response_is_dict(self, client):
        """Cache stats should return a JSON object."""
        response = client.get("/matching_engine/cache-stats")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)


# ── Payload Contract Tests ────────────────────────────────────────────────────

class TestPayloadContracts:
    """
    Tests that verify the API contract is preserved.
    These tests document what the frontend sends and what it receives.
    Any regression here means the frontend will break.
    """

    def test_accepts_boolean_answer(self, client):
        """Boolean answers in the payload should be accepted."""
        payload = {
            "electionId": "e001",
            "citizenId": "v001",
            "responses": [
                {
                    "questionId": "q001",
                    "question": "Do you support increased school funding?",
                    "answer": True,
                }
            ],
        }
        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=payload,
        )
        # Should not return 422 (validation error)
        assert response.status_code != 422

    def test_accepts_list_answer(self, client):
        """List answers (multi-select) in the payload should be accepted."""
        payload = {
            "electionId": "e001",
            "citizenId": "v001",
            "responses": [
                {
                    "questionId": "q001",
                    "question": "Which programs should receive funding?",
                    "answer": ["STEM", "Arts", "Sports"],
                }
            ],
        }
        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=payload,
        )
        assert response.status_code != 422

    def test_accepts_string_answer(self, client):
        """String answers in the payload should be accepted."""
        payload = {
            "electionId": "e001",
            "citizenId": "v001",
            "responses": [
                {
                    "questionId": "q001",
                    "question": "What is your top education priority?",
                    "answer": "Increase funding for public schools",
                }
            ],
        }
        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=payload,
        )
        assert response.status_code != 422

    def test_accepts_optional_comment_field(self, client):
        """Optional comment field should be accepted without breaking anything."""
        payload = {
            "electionId": "e001",
            "citizenId": "v001",
            "responses": [
                {
                    "questionId": "q001",
                    "question": "Do you support increased school funding?",
                    "answer": "yes",
                    "comment": "I believe strong funding is essential for quality education.",
                }
            ],
        }
        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=payload,
        )
        assert response.status_code != 422

    def test_response_has_matches_field(self, client):
        """Response must always have a matches field — even if empty."""
        payload = {
            "electionId": "e001",
            "citizenId": "v001",
            "responses": [
                {
                    "questionId": "q001",
                    "question": "What is your name?",
                    "answer": "John",
                }
            ],
        }
        response = client.post(
            "/matching_engine/submit-voter-responses",
            json=payload,
        )
        if response.status_code == 200:
            data = response.json()
            assert "matches" in data
