import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app.main import create_app
from app.schemas.voters_schema import (
    VoterSubmissionSchema,
    MatchResultsResponseSchema,
    CandidateMatchSchema,
    IssueMatchDetailSchema
)


@pytest.fixture
def test_app():
    """Create a test FastAPI app."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_voter_submission():
    """Create a sample voter submission for testing."""
    return {
        "election_id": "e001",
        "citizen_id": "v001",
        "responses": [
            {
                "question_id": "q001",
                "question": "Should your neighborhood students have access to a language immersion middle school within a 30-minute commute?",
                "answer": "Strongly Agree",
                "category": "Education Access"
            },
            {
                "question_id": "q002",
                "question": "Which educational programs should receive increased funding? (Select all that apply)",
                "answer": ["STEM initiatives", "Special education", "Arts and music"],
                "category": "School Funding"
            },
            {
                "question_id": "q003",
                "question": "Do you think it's essential for the your neighborhood council member to prioritize mental health resources for students?",
                "answer": "Strongly Agree",
                "category": "Student Support"
            }
        ],
        "completed_at": "2025-05-13T09:30:45.123Z"
    }


@pytest.fixture
def sample_match_result():
    """Create a sample match result for mocking."""
    return MatchResultsResponseSchema(
        voter_id="v001",
        election_id="e001",
        matches=[
            CandidateMatchSchema(
                candidate_id="c001",
                candidate_name="Jane Smith",
                candidate_title="Senate Candidate",
                match_percentage=85,
                top_aligned_issues=["Education Access", "Student Support"],
                issue_matches=[
                    IssueMatchDetailSchema(
                        issue="Education Access",
                        alignment="Strongly Aligned",
                        voter_position="Strongly Agree",
                        candidate_position="Strongly Agree"
                    ),
                    IssueMatchDetailSchema(
                        issue="Student Support",
                        alignment="Strongly Aligned",
                        voter_position="Strongly Agree",
                        candidate_position="Strongly Agree"
                    )
                ]
            ),
            CandidateMatchSchema(
                candidate_id="c002",
                candidate_name="Michael Johnson",
                candidate_title="Senate Candidate",
                match_percentage=65,
                top_aligned_issues=["Student Support"],
                issue_matches=[
                    IssueMatchDetailSchema(
                        issue="Education Access",
                        alignment="Weakly Aligned",
                        voter_position="Strongly Agree",
                        candidate_position="Disagree"
                    ),
                    IssueMatchDetailSchema(
                        issue="Student Support",
                        alignment="Moderately Aligned",
                        voter_position="Strongly Agree",
                        candidate_position="Agree"
                    )
                ]
            )
        ]
    )


class TestMatchingEngineRouter:
    """Test the matching engine router."""

    @patch('app.api.routes.matching_engine.matching_engine.process_voter_submission', new_callable=AsyncMock)
    def test_submit_voter_responses(self, mock_process_submission, test_app, sample_voter_submission, sample_match_result):
        """Test the submit_voter_responses endpoint with valid data."""
        # Mock the process_voter_submission method to return our sample result
        mock_process_submission.return_value = sample_match_result

        # Make the request
        response = test_app.post("/api/v1/matching_engine", json=sample_voter_submission)

        # Verify the response
        assert response.status_code == 200
        result = response.json()

        # Check result structure
        assert result["voter_id"] == sample_match_result.voter_id
        assert result["election_id"] == sample_match_result.election_id
        assert len(result["matches"]) == len(sample_match_result.matches)

        # Check match details
        assert result["matches"][0]["candidate_id"] == "c001"
        assert result["matches"][0]["match_percentage"] == 85
        assert "Education Access" in result["matches"][0]["top_aligned_issues"]

        # Verify that the mock was called with the right data
        mock_process_submission.assert_called_once()
        # We need to extract the argument to check it's the correct schema
        call_args = mock_process_submission.call_args[0][0]
        assert isinstance(call_args, VoterSubmissionSchema)
        assert call_args.election_id == sample_voter_submission["election_id"]
        assert call_args.citizen_id == sample_voter_submission["citizen_id"]
        assert len(call_args.responses) == len(sample_voter_submission["responses"])

    @patch('app.api.routes.matching_engine.matching_engine.process_voter_submission')
    def test_submit_voter_responses_error(self, mock_process_submission, test_app, sample_voter_submission):
        """Test the submit_voter_responses endpoint when an error occurs."""
        # Mock an exception being raised
        mock_process_submission.side_effect = Exception("Test error")

        # Make the request
        response = test_app.post("/api/v1/matching_engine", json=sample_voter_submission)

        # Verify the error response
        assert response.status_code == 500
        result = response.json()
        assert "detail" in result
        assert "Test error" in result["detail"]

    def test_submit_voter_responses_invalid_data(self, test_app):
        """Test the submit_voter_responses endpoint with invalid data."""
        # Missing required fields
        invalid_data = {
            "election_id": "e001",
            # Missing citizen_id
            "responses": []
        }

        # Make the request
        response = test_app.post("/api/v1/matching_engine", json=invalid_data)

        # Verify the validation error response
        assert response.status_code == 422
        result = response.json()
        assert "detail" in result
