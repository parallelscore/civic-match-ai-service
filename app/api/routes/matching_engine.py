from datetime import datetime
from fastapi import HTTPException, status

from app.utils.logging_util import setup_logger
from app.api.routes.base_router import RouterManager
from app.schemas.voters_schema import VoterSubmissionSchema
from app.services.candidate_service import candidate_service
from app.services.enhanced_matching_engine_service import enhanced_matching_engine


class MatchingEngineRouter:
    """Router for the enhanced matching engine API endpoints."""

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

        # Add a health check endpoint for the matching engine
        self.router_manager.add_route(
            path="/matching_engine/health",
            handler_method=self.health_check,
            methods=["GET"],
            tags=["Matching Engine"],
            status_code=status.HTTP_200_OK
        )

        # Add cache stats endpoint
        self.router_manager.add_route(
            path="/matching_engine/cache/stats",
            handler_method=self.cache_stats,
            methods=["GET"],
            tags=["Matching Engine"],
            status_code=status.HTTP_200_OK
        )

        self.router_manager.add_route(
            path="/matching_engine/debug",
            handler_method=self.debug_matching,
            methods=["POST"],
            tags=["Matching Engine Debug"],
            status_code=status.HTTP_200_OK
        )

    async def submit_voter_responses(self, submission: VoterSubmissionSchema):
        """
        Submit voter responses and calculate enhanced matches.

        Args:
            submission: The voter's submission containing responses to questions

        Returns:
            Enhanced match results for all candidates with detailed explanations
        """
        try:
            self.logger.info(f"Received enhanced match request for voter {submission.citizen_id} in election {submission.election_id}")

            # Process submission using enhanced matching engine
            results = await enhanced_matching_engine.process_voter_submission(submission)

            self.logger.info(f"Generated {len(results.matches)} matches for voter {submission.citizen_id} using {results.processing_method}")

            return results

        except Exception as e:
            self.logger.error(f"Error processing voter submission: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing submission: {str(e)}"
            )

    async def health_check(self):
        """
        Health check endpoint for the matching engine services.
        """
        try:
            from app.services.caching_service import cache_service
            from app.services.semantic_matching_service import semantic_service
            from app.services.llm_service import llm_service
            from app.core.config import settings

            health_status = {
                "status": "healthy",
                "timestamp": str(datetime.now()),
                "services": {
                    "semantic_matching": {
                        "enabled": settings.ENABLE_SEMANTIC_MATCHING,
                        "model_loaded": semantic_service.model is not None,
                        "model_name": settings.EMBEDDING_MODEL
                    },
                    "llm_matching": {
                        "enabled": settings.ENABLE_LLM_MATCHING,
                        "provider": settings.LLM_PROVIDER,
                        "model": settings.LLM_MODEL,
                        "client_initialized": llm_service.client is not None
                    },
                    "caching": {
                        "cache_type": "redis" if cache_service.redis_client else "memory",
                        "redis_connected": cache_service.redis_client is not None
                    }
                },
                "configuration": {
                    "embedding_threshold": settings.EMBEDDING_SIMILARITY_THRESHOLD,
                    "cache_ttl": settings.CACHE_TTL_SECONDS,
                    "llm_timeout": settings.LLM_TIMEOUT_SECONDS
                }
            }

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Health check failed"
            )

    async def cache_stats(self):
        """
        Get cache performance statistics.
        """
        try:
            from app.services.caching_service import cache_service

            stats = cache_service.get_cache_stats()
            return {
                "cache_stats": stats,
                "timestamp": str(datetime.now())
            }

        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve cache statistics"
            )

    async def debug_matching(self, submission: VoterSubmissionSchema):
        """
        Debug endpoint to analyze what's happening in the matching process.
        """
        try:
            self.logger.info(f"DEBUG: Analyzing matching for voter {submission.citizen_id}")

            # Get candidates
            candidates = await candidate_service.get_candidates_for_election(submission.election_id)

            if not candidates:
                return {"error": "No candidates found"}

            debug_info = {
                "voter_questions": [r.question for r in submission.responses],
                "candidates": []
            }

            for candidate in candidates:
                candidate_info = {
                    "candidate_id": candidate.candidate_id,
                    "candidate_name": getattr(candidate, "name", "No name"),
                    "candidate_questions": [r.question for r in candidate.responses],
                    "semantic_matches": []
                }

                # Test semantic matching
                voter_responses = {r.question: r for r in submission.responses}
                candidate_responses = {r.question: r for r in candidate.responses}

                voter_questions = list(voter_responses.keys())
                candidate_questions = list(candidate_responses.keys())

                # Get semantic similarities
                from app.services.semantic_matching_service import semantic_service
                batch_results = semantic_service.batch_find_similar_questions(
                    voter_questions, candidate_questions
                )

                for voter_q, similarities in batch_results.items():
                    if similarities:
                        best_match = similarities[0]
                        candidate_info["semantic_matches"].append({
                            "voter_question": voter_q,
                            "candidate_question": best_match.candidate_question,
                            "similarity_score": best_match.similarity_score,
                            "voter_answer": voter_responses[voter_q].answer,
                            "candidate_answer": candidate_responses[best_match.candidate_question].answer
                        })

                debug_info["candidates"].append(candidate_info)

            return debug_info

        except Exception as e:
            self.logger.error(f"Debug matching failed: {str(e)}")
            return {"error": str(e)}