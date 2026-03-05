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
        Get all candidates for a specific election.
        Returns all candidates regardless of completion status - the matching engine
        will handle incomplete candidates by giving them 0% match.

        Args:
            election_id: The ID of the election

        Returns:
            List of all candidates with their responses
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

                    self.logger.info(f"Retrieved {len(candidates_data)} candidate records")

                    # Process all candidates (no filtering based on completion status)
                    # New lenient eligibility: candidates with any responses can be matched
                    candidates = []
                    matchable_count = 0
                    no_response_count = 0

                    for candidate_data in candidates_data:
                        # Check response availability for logging purposes
                        has_completed_profile = candidate_data.get("hasCompletedProfile", False)
                        has_completed_questionnaire = candidate_data.get("hasCompletedQuestionnaire", False)
                        responses = candidate_data.get("responses", [])
                        response_count = len(responses)

                        # New eligibility: only requires responses, ignores completion flags
                        is_matchable = response_count > 0

                        if is_matchable:
                            matchable_count += 1
                            # Log if flags are false but we still have data (common scenario)
                            if not has_completed_questionnaire and response_count > 0:
                                self.logger.debug(
                                    f"Candidate {candidate_data.get('candidateId', 'unknown')} has {response_count} responses "
                                    f"(questionnaire flag={has_completed_questionnaire}) - will attempt matching"
                                )
                            else:
                                self.logger.debug(
                                    f"Candidate {candidate_data.get('candidateId', 'unknown')} is matchable with {response_count} responses"
                                )
                        else:
                            no_response_count += 1
                            self.logger.debug(
                                f"Candidate {candidate_data.get('candidateId', 'unknown')} has no responses - will receive 0% match. "
                                f"Profile: {has_completed_profile}, Questionnaire: {has_completed_questionnaire}"
                            )

                        # Process all candidates regardless of completion status
                        # Handle the candidate data structure from your ward8_election_data.json
                        # The data uses camelCase instead of snake_case for various fields

                        # Convert candidateId to candidate_id
                        if 'candidateId' in candidate_data:
                            candidate_data['candidate_id'] = candidate_data.pop('candidateId')

                        # Convert electionId to election_id
                        if 'electionId' in candidate_data:
                            candidate_data['election_id'] = candidate_data.pop('electionId')

                        # Convert completion status fields from camelCase to snake_case
                        if 'hasCompletedProfile' in candidate_data:
                            candidate_data['has_completed_profile'] = candidate_data.pop('hasCompletedProfile')

                        if 'hasCompletedQuestionnaire' in candidate_data:
                            candidate_data['has_completed_questionnaire'] = candidate_data.pop('hasCompletedQuestionnaire')

                        # Handle responses that might use 'electionId' instead of 'election_id'
                        if 'responses' in candidate_data:
                            for response in candidate_data['responses']:
                                if 'electionId' in response:
                                    response['election_id'] = response.pop('electionId')

                        try:
                            candidate = CandidateResponseSchema.model_validate(candidate_data)
                            candidates.append(candidate)
                        except Exception as validation_error:
                            # Log prominently — a dropped candidate will be invisible
                            # to the voter and could cause confusion on the frontend.
                            candidate_id = candidate_data.get(
                                "candidate_id",
                                candidate_data.get("candidateId", "unknown")
                            )
                            self.logger.error(
                                f"Candidate '{candidate_id}' failed schema validation and "
                                f"will NOT appear in results (not even as UNMATCH). "
                                f"Reason: {validation_error}. "
                                f"Raw data keys: {list(candidate_data.keys())}"
                            )
                            continue

                    self.logger.info(
                        f"Successfully processed {len(candidates)} total candidates "
                        f"({matchable_count} with responses for matching, {no_response_count} without responses)"
                    )
                    return candidates

        except Exception as e:
            self.logger.error(f"Error in get_candidates_for_election: {str(e)}")
            return []


# Create an instance for dependency injection
candidate_service = CandidateService()
