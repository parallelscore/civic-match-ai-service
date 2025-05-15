from datetime import datetime
from typing import Any, Tuple

from app.schemas.candidate_schema import CandidateResponseSchema
from app.schemas.match_result_schema import MatchResultSchema, MatchResultsResponseSchema, MatchResultDetailSchema
from app.schemas.voters_schema import VoterSubmissionSchema
from app.services.candidate_service import candidate_service
from app.utils.logging_util import setup_logger


class MatchingEngine:
    """
    Core matching engine service that implements the matching algorithm.

    This engine:
    1. Processes voter responses
    2. Retrieves candidate responses from the candidate service
    3. Calculates match scores
    4. Generates and returns match results
    """

    def __init__(self):
        self.logger = setup_logger(__name__)

    async def process_voter_submission(self, submission: VoterSubmissionSchema) -> MatchResultsResponseSchema:
        """
        Process a voter's submission of responses and generate match results.

        Args:
            submission: The voter's submission containing responses to questions

        Returns:
            Match results for all candidates
        """
        self.logger.info(
            f"Processing submission for voter {submission.citizen_id} in election {submission.election_id}")

        # Step 1: Get candidates for the election
        candidates = await candidate_service.get_candidates_for_election(submission.election_id)
        self.logger.info(f"Retrieved {len(candidates)} candidates for election {submission.election_id}")

        if not candidates:
            self.logger.warning(f"No candidates found for election {submission.election_id}")
            return MatchResultsResponseSchema(
                voter_id=submission.citizen_id,
                election_id=submission.election_id,
                matches=[],
                generated_at=datetime.now()
            )

        # Step 2: Calculate matches for each candidate
        match_results = []
        for candidate in candidates:
            match_result = self._calculate_match(submission, candidate)
            match_results.append(match_result)

        # Step 3: Sort matches by overall score (descending)
        match_results.sort(key=lambda x: x.overall_score, reverse=True)

        return MatchResultsResponseSchema(
            voter_id=submission.citizen_id,
            election_id=submission.election_id,
            matches=match_results,
            generated_at=datetime.now()
        )

    def _calculate_match(self, voter_submission: VoterSubmissionSchema,
                         candidate: CandidateResponseSchema) -> MatchResultSchema:
        """
        Calculate a match between a voter and a candidate.

        Args:
            voter_submission: The voter's submission
            candidate: The candidate's information and responses

        Returns:
            Match result between the voter and candidate
        """
        # Organization by question
        voter_responses = {r.question: r for r in voter_submission.responses}

        self.logger.info(f"Voter responses: {voter_responses}")

        candidate_responses = {r.question: r for r in candidate.responses}

        self.logger.info(f"Candidate responses: {candidate_responses}")

        # Find common questions
        common_questions = set(voter_responses.keys()).intersection(set(candidate_responses.keys()))

        self.logger.info(f"Common questions: {common_questions}")
        self.logger.info(
            f"Found {len(common_questions)} common questions between voter and candidate {candidate.candidate_id}")

        if not common_questions:
            # No common questions, no match
            return MatchResultSchema(
                candidate_id=candidate.candidate_id,
                overall_score=0.0,
                details=[],
                top_aligned_issues=[],
                top_misaligned_issues=[]
            )

        # Group by categories (using the question text as a category for now)
        categories = {}
        scores_by_question = {}

        for question in common_questions:
            voter_answer = voter_responses[question].answer
            candidate_answer = candidate_responses[question].answer

            # Calculate similarity for this question
            similarity, explanation = self._calculate_similarity(
                voter_answer,
                candidate_answer,
                self._determine_response_type(voter_answer)
            )

            category = "General"  # Default category

            # Initialize a category if needed
            if category not in categories:
                categories[category] = {
                    "scores": [],
                    "details": []
                }

            # Add score to category
            categories[category]["scores"].append(similarity)

            # Add detail for this question
            categories[category]["details"].append({
                "question": question,
                "voter_answer": voter_answer,
                "candidate_answer": candidate_answer,
                "score": similarity,
                "explanation": explanation
            })

            # Keep track of scores by question for top issues
            scores_by_question[question] = similarity

        # Calculate category scores
        category_details = []
        for category, data in categories.items():
            avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0

            # Determine alignment level
            if avg_score >= 0.7:
                alignment = "high"
            elif avg_score >= 0.4:
                alignment = "medium"
            else:
                alignment = "low"

            category_details.append(
                MatchResultDetailSchema(
                    category=category,
                    score=avg_score,
                    alignment=alignment,
                    key_issues=data["details"]
                )
            )

        # Calculate overall score as average of all scores
        all_scores = [score for data in categories.values() for score in data["scores"]]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0

        # Find top aligned and misaligned issues
        sorted_questions = sorted(scores_by_question.items(), key=lambda x: x[1], reverse=True)
        top_aligned = [q for q, s in sorted_questions[:3] if s >= 0.6]
        top_misaligned = [q for q, s in sorted_questions[-3:] if s < 0.4]

        return MatchResultSchema(
            candidate_id=candidate.candidate_id,
            overall_score=overall_score,
            details=category_details,
            top_aligned_issues=top_aligned,
            top_misaligned_issues=top_misaligned
        )

    def _determine_response_type(self, answer: Any) -> str:
        """Determine the response type based on the answer."""
        if isinstance(answer, bool):
            return 'binary'
        elif isinstance(answer, list):
            return 'multiple-choice'
        elif isinstance(answer, dict):
            return 'ranking'
        else:
            return 'text'

    def _calculate_similarity(self, voter_answer: Any, candidate_answer: Any, response_type: str) -> Tuple[float, str]:
        """
        Calculate similarity between voter and candidate answers.

        Returns:
            Tuple of (similarity_score, explanation)
        """
        if response_type == 'binary':
            # Direct binary comparison
            similarity = 1.0 if voter_answer == candidate_answer else 0.0
            explanation = "Exact match" if similarity == 1.0 else "Different responses"

        elif response_type == 'multiple-choice':
            # Calculate Jaccard similarity for multiple choice
            voter_set = set(voter_answer if isinstance(voter_answer, list) else [voter_answer])
            candidate_set = set(candidate_answer if isinstance(candidate_answer, list) else [candidate_answer])

            if not voter_set or not candidate_set:
                similarity = 0.0
                explanation = "One or both responses are empty"
            else:
                intersection = len(voter_set.intersection(candidate_set))
                union = len(voter_set.union(candidate_set))
                similarity = intersection / union if union > 0 else 0.0

                if similarity == 1.0:
                    explanation = "Exact match on all selections"
                elif similarity > 0:
                    explanation = f"Partial match: {intersection} common selections out of {union} total"
                else:
                    explanation = "No common selections"

        elif response_type == 'ranking':
            # Handle ranking comparison
            # For MVP, we'll use a simplified approach to comparing top choices
            voter_ranking = voter_answer
            candidate_ranking = candidate_answer

            # Get top 3 choices from each ranking
            voter_top = list(voter_ranking.keys())[:3] if isinstance(voter_ranking, dict) else []
            candidate_top = list(candidate_ranking.keys())[:3] if isinstance(candidate_ranking, dict) else []

            # Calculate similarity based on the presence and position of items
            common_items = set(voter_top).intersection(set(candidate_top))
            similarity = len(common_items) / max(len(voter_top), len(candidate_top))

            if similarity == 1.0 and voter_top == candidate_top:
                explanation = "Exact match on priorities"
            elif similarity > 0:
                explanation = f"Partial match: {len(common_items)} common priorities"
            else:
                explanation = "Different priorities"

        else:  # text responses
            # For text responses, we use a simple approach for MVP
            # In a production system; this would use NLP techniques
            voter_text = str(voter_answer).lower()
            candidate_text = str(candidate_answer).lower()

            # Simple text similarity: % of words in common
            voter_words = set(voter_text.split())
            candidate_words = set(candidate_text.split())

            if not voter_words or not candidate_words:
                similarity = 0.0
                explanation = "One or both responses are empty"
            else:
                common_words = voter_words.intersection(candidate_words)
                all_words = voter_words.union(candidate_words)
                similarity = len(common_words) / len(all_words)

                if similarity >= 0.7:
                    explanation = "High text similarity"
                elif similarity >= 0.3:
                    explanation = "Moderate text similarity"
                else:
                    explanation = "Low text similarity"

        return similarity, explanation


# Create an instance for dependency injection
matching_engine = MatchingEngine()
