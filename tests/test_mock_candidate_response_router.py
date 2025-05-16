import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def test_app():
    """Create a test FastAPI app."""
    app = create_app()
    return TestClient(app)


class TestMockCandidatesResponseRouter:
    """Test suite for the mock candidates response router."""

    def test_get_mock_candidates_response(self, test_app):
        """Test the mock candidates endpoint with a valid election ID."""
        # Make the request with a test election ID
        response = test_app.get("/api/v1/recommendations/elections/test-election/candidates")

        # Verify the response
        assert response.status_code == 200
        candidates = response.json()

        # Check the structure and content of the response
        assert len(candidates) == 3  # Should return 3 mock candidates

        # Check the first candidate
        assert candidates[0]["candidate_id"] == "c001"
        assert candidates[0]["name"] == "Jane Smith"
        assert candidates[0]["election_id"] == "test-election"  # Should match the requested election ID
        assert len(candidates[0]["responses"]) > 0

        # Check response fields in the first candidate's first response
        first_response = candidates[0]["responses"][0]
        assert "id" in first_response
        assert "question" in first_response
        assert "answer" in first_response
        assert "comment" in first_response
        assert first_response["election_id"] == "test-election"  # Should match the requested election ID

        # Check the second candidate
        assert candidates[1]["candidate_id"] == "c002"
        assert candidates[1]["name"] == "Michael Johnson"

        # Check the third candidate
        assert candidates[2]["candidate_id"] == "c003"
        assert candidates[2]["name"] == "Aisha Washington"

    def test_get_mock_candidates_response_different_election_id(self, test_app):
        """Test the mock candidates endpoint with a different election ID."""
        # Make the request with a different election ID
        response = test_app.get("/api/v1/recommendations/elections/another-election/candidates")

        # Verify the response
        assert response.status_code == 200
        candidates = response.json()

        # Verify that the election ID in the response matches the requested ID
        assert candidates[0]["election_id"] == "another-election"
        assert candidates[0]["responses"][0]["election_id"] == "another-election"