import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.candidate_service import CandidateService
from app.schemas.candidate_schema import CandidateResponseSchema


@pytest.fixture
def candidate_service():
    """Create a CandidateService instance for testing."""
    return CandidateService()


@pytest.fixture
def sample_candidate_response():
    """Create a sample API response for candidates."""
    return [
        {
            "candidate_id": "c001",
            "name": "Jane Smith",
            "election_id": "e001",
            "responses": [
                {
                    "id": "r001",
                    "question": "Should your neighborhood students have access to a language immersion middle school within a 30-minute commute?",
                    "answer": "Strongly Agree",
                    "comment": "Language immersion programs are crucial for our students' future success in a global economy.",
                    "election_id": "e001"
                },
                {
                    "id": "r002",
                    "question": "Which educational programs should receive increased funding? (Select all that apply)",
                    "answer": ["STEM initiatives", "Arts and music", "Special education"],
                    "comment": "We need balanced funding across multiple educational areas.",
                    "election_id": "e001"
                }
            ]
        },
        {
            "candidate_id": "c002",
            "name": "Michael Johnson",
            "election_id": "e001",
            "responses": [
                {
                    "id": "r009",
                    "question": "Should your neighborhood students have access to a language immersion middle school within a 30-minute commute?",
                    "answer": "Disagree",
                    "comment": "We should focus on core academics before expanding to immersion programs.",
                    "election_id": "e001"
                }
            ]
        }
    ]


class TestCandidateService:
    """Test suite for the candidate service."""

    @pytest.mark.asyncio
    @patch('app.services.candidate_service.aiohttp.ClientSession')
    async def test_get_candidates_for_election_success(self, mock_session, candidate_service, sample_candidate_response):
        """Test getting candidates with a successful API response."""
        # Setup the mock session and response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=sample_candidate_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session_instance = MagicMock()
        mock_session_instance.get = MagicMock(return_value=mock_response)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock()

        mock_session.return_value = mock_session_instance

        # Call the method
        result = await candidate_service.get_candidates_for_election("e001")

        # Verify the result
        assert len(result) == 2
        assert isinstance(result[0], CandidateResponseSchema)
        assert result[0].candidate_id == "c001"
        assert result[1].candidate_id == "c002"
        assert len(result[0].responses) == 2
        assert len(result[1].responses) == 1

        # Check the URL construction
        expected_url = f"{candidate_service.base_url}/recommendations/elections/e001/candidates"
        mock_session_instance.get.assert_called_once_with(expected_url)

    @pytest.mark.asyncio
    @patch('app.services.candidate_service.aiohttp.ClientSession')
    async def test_get_candidates_for_election_non_200_response(self, mock_session, candidate_service):
        """Test getting candidates with a non-200 API response."""
        # Setup the mock session and response
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session_instance = MagicMock()
        mock_session_instance.get = MagicMock(return_value=mock_response)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock()

        mock_session.return_value = mock_session_instance

        # Call the method
        result = await candidate_service.get_candidates_for_election("e001")

        # Verify the result is an empty list on error
        assert result == []

    @pytest.mark.asyncio
    @patch('app.services.candidate_service.aiohttp.ClientSession')
    async def test_get_candidates_for_election_exception(self, mock_session, candidate_service):
        """Test getting candidates when an exception occurs."""
        # Setup the mock session to raise an exception
        mock_session_instance = MagicMock()
        mock_session_instance.__aenter__ = AsyncMock(side_effect=Exception("Connection error"))
        mock_session_instance.__aexit__ = AsyncMock()

        mock_session.return_value = mock_session_instance

        # Call the method
        result = await candidate_service.get_candidates_for_election("e001")

        # Verify the result is an empty list on error
        assert result == []

    @patch('app.services.candidate_service.settings')
    def test_init_with_real_api(self, mock_settings):
        """Test initialization with real API config."""
        # Configure mock settings
        mock_settings.USE_MOCK_BACKEND_API_URL = False
        mock_settings.BACKEND_API_URL = "https://real-api.example.com"

        # Create a new service instance
        service = CandidateService()

        # Verify the base URL is set correctly
        assert service.base_url == "https://real-api.example.com"

    @patch('app.services.candidate_service.settings')
    def test_init_with_mock_api(self, mock_settings):
        """Test initialization with mock API config."""
        # Configure mock settings
        mock_settings.USE_MOCK_BACKEND_API_URL = True
        mock_settings.MOCK_BACKEND_API_URL = "http://localhost:8000"

        # Create a new service instance
        service = CandidateService()

        # Verify the base URL is set correctly
        assert service.base_url == "http://localhost:8000"