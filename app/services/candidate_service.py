from typing import List

import aiohttp

from app.core.config import settings
from app.schemas.candidate_schema import CandidateResponseSchema, CandidateResponseItemSchema
from app.utils.logging_util import setup_logger


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
            self.logger.info(f"Using real API at: {self.base_url}")  # Configure this in settings

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
                url = f"{self.base_url}/recommendations/elections/{election_id}/candidates"
                self.logger.info(f"Fetching candidates from: {url}")

                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"Error fetching candidates: {response.status} {await response.text()}")
                        return []

                    data = await response.json()
                    self.logger.info(f"Received {len(data)} candidates")

                    # Convert to our schema
                    candidates = []
                    for candidate_data in data:
                        candidate = CandidateResponseSchema(
                            candidate_id=candidate_data.get("candidate_id"),
                            election_id=candidate_data.get("election_id"),
                            responses=[
                                CandidateResponseItemSchema(
                                    id=response.get("id"),
                                    question=response.get("question"),
                                    answer=response.get("answer"),
                                    comment=response.get("comment"),
                                    election_id=response.get("election_id")
                                )
                                for response in candidate_data.get("responses", [])
                            ]
                        )
                        candidates.append(candidate)

                    return candidates

        except Exception as e:
            self.logger.error(f"Error in get_candidates_for_election: {str(e)}")
            return []


# Create an instance for dependency injection
candidate_service = CandidateService()


# Example usage:
async def main():
    service = CandidateService()
    candidates = await service.get_candidates_for_election("election_id")
    print(candidates)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
