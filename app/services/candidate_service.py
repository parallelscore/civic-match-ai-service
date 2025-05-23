import aiohttp
from typing import List

from app.core.config import settings
from app.utils.logging_util import setup_logger
from app.schemas.candidate_schema import CandidateResponseSchema


class CandidateService:
    """Service for interacting with the candidate API."""

    def __init__(self):
        self.logger = setup_logger(__name__)
        self.base_url = settings.BACKEND_API_URL

        # Determine which API URL to use based on configuration
        if settings.USE_MOCK_BACKEND_API_URL:
            self.base_url = settings.MOCK_BACKEND_API_URL
            self.logger.info(f"Using mock API at: {self.base_url}")
        else:
            self.base_url = settings.BACKEND_API_URL
            self.logger.info(f"Using real API at: {self.base_url}")

    async def get_candidates_for_election(self, election_id: str) -> List[CandidateResponseSchema]:
        """
        Get all candidates and their responses for a specific election.

        Args:
            election_id: The ID of the election

        Returns:
            List of candidates with their responses
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/candidates/recommendation/{election_id}"
                self.logger.info(f"Fetching candidates from: {url}")

                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"Error fetching candidates: {response.status} {await response.text()}")
                        return []

                    data = await response.json()

                    # Handle both direct array and nested data structure
                    if isinstance(data, list):
                        candidates_data = data
                    else:
                        candidates_data = data.get("data", [])

                    self.logger.info(f"Processing {len(candidates_data)} candidates")

                    candidates = []
                    for candidate_data in candidates_data:
                        # Handle the candidate data structure from your ward8_election_data.json
                        # The data uses 'candidateId' instead of 'candidate_id'
                        if 'candidateId' in candidate_data:
                            candidate_data['candidate_id'] = candidate_data.pop('candidateId')

                        if 'electionId' in candidate_data:
                            candidate_data['election_id'] = candidate_data.pop('electionId')

                        # Handle responses that might use 'electionId' instead of 'election_id'
                        if 'responses' in candidate_data:
                            for response in candidate_data['responses']:
                                if 'electionId' in response:
                                    response['election_id'] = response.pop('electionId')

                        try:
                            candidate = CandidateResponseSchema.model_validate(candidate_data)
                            candidates.append(candidate)
                        except Exception as validation_error:
                            self.logger.error(f"Failed to validate candidate data: {validation_error}")
                            self.logger.debug(f"Problematic candidate data: {candidate_data}")
                            continue

                    self.logger.info(f"Successfully processed {len(candidates)} candidates")
                    return candidates

        except Exception as e:
            self.logger.error(f"Error in get_candidates_for_election: {str(e)}")
            return []


# Create an instance for dependency injection
candidate_service = CandidateService()