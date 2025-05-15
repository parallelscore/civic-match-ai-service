from fastapi import HTTPException, status

from app.api.routes.base_router import RouterManager
from app.schemas.voters_schema import VoterSubmissionSchema
from app.services.matching_engine_service import matching_engine
from app.utils.logging_util import setup_logger


class MatchingEngineRouter:
    """Router for the matching engine API endpoints."""

    def __init__(self):
        self.router_manager = RouterManager()
        self.logger = setup_logger(__name__)

        # Register routes
        self.router_manager.add_route(
            path="/matching_engine",
            handler_method=self.submit_voter_responses,
            methods=["POST"],
            tags=["Matching Engine"],
            status_code=status.HTTP_200_OK
        )

    async def submit_voter_responses(self, submission: VoterSubmissionSchema):
        """
        Submit voter responses and calculate matches.

        Args:
            submission: The voter's submission containing responses to questions

        Returns:
            Match results for all candidates
        """
        try:
            self.logger.info(f"Received match request for voter {submission.election_id}")
            results = await matching_engine.process_voter_submission(submission)
            return results

        except Exception as e:
            self.logger.error(f"Error processing voter submission: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing submission: {str(e)}"
            )
