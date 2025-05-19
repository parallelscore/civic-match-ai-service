import math
from datetime import datetime
from typing import Any, Tuple, Dict, List

from app.utils.logging_util import setup_logger
from app.schemas.voters_schema import VoterSubmissionSchema
from app.services.candidate_service import candidate_service
from app.schemas.candidate_schema import CandidateResponseSchema
from app.schemas.voters_schema import (CandidateMatchSchema, MatchResultsResponseSchema, IssueMatchDetailSchema,
                                       IssuePrioritySchema, WeightedIssueSchema)


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
        match_results.sort(key=lambda x: x.match_percentage, reverse=True)

        return MatchResultsResponseSchema(
            voter_id=submission.citizen_id,
            election_id=submission.election_id,
            matches=match_results,
            generated_at=datetime.now()
        )

    def _calculate_match(self, voter_submission: VoterSubmissionSchema,
                         candidate: CandidateResponseSchema) -> CandidateMatchSchema:
        """
        Calculate a match between a voter and a candidate.
        Works with or without explicit priorities.

        Args:
            voter_submission: The voter's submission with responses and optional priorities
            candidate: The candidate's information and responses

        Returns:
            Match result between the voter and candidate
        """
        # Organization by question
        voter_responses = {r.question: r for r in voter_submission.responses}
        candidate_responses = {r.question: r for r in candidate.responses}

        # Find common questions
        common_questions = set(voter_responses.keys()).intersection(set(candidate_responses.keys()))
        self.logger.info(
            f"Found {len(common_questions)} common questions between voter and candidate {candidate.candidate_id}")

        if not common_questions:
            # No common questions, no match
            return CandidateMatchSchema(
                candidate_id=candidate.candidate_id,
                candidate_name=getattr(candidate, "name", candidate.candidate_id),
                candidate_title="Candidate",
                match_percentage=0,
                top_aligned_issues=[],
                issue_matches=[],
                weighted_issues=[],
                strongest_match_factors=[]
            )

        # Categorize questions by issue/topic
        issue_categories = self._categorize_questions(common_questions, voter_responses, candidate_responses)

        # Extract or infer priorities
        category_weights = self._extract_category_weights(voter_submission)

        # Track if we're using explicit priorities or inferred ones
        using_explicit_priorities = (
                hasattr(voter_submission, 'issue_priorities') and
                voter_submission.issue_priorities is not None and
                len(voter_submission.issue_priorities) > 0
        )

        # Calculate issue match details and scores
        issue_matches = []
        issue_scores = {}
        weighted_scores = []
        total_weight = 0
        weighted_issues = []
        score_contributions = {}

        for issue, questions in issue_categories.items():
            # Get the weight for this category (default to 1.0 if not specified)
            weight = category_weights.get(issue, 1.0)
            total_weight += weight

            # Calculate the average score for this issue
            scores = []
            for question in questions:
                voter_answer = voter_responses[question].answer
                candidate_answer = candidate_responses[question].answer

                similarity, _ = self._calculate_similarity(
                    voter_answer,
                    candidate_answer,
                    self._determine_response_type(voter_answer)
                )
                scores.append(similarity)

            # Calculate issue score
            avg_score = sum(scores) / len(scores) if scores else 0
            issue_scores[issue] = avg_score

            # Add weighted score contribution
            weighted_contribution = avg_score * weight
            weighted_scores.append(weighted_contribution)

            # Track weighted scores for explanation
            impact_score = weighted_contribution / total_weight if total_weight > 0 else 0
            score_contributions[issue] = impact_score

            # Record weighted issue data
            weighted_issues.append({
                "category": issue,
                "weight": weight,
                "impact_score": impact_score
            })

            # Get an alignment level
            alignment = self._get_alignment_level(avg_score)

            # For display purposes, select the first question's responses
            sample_question = questions[0]
            voter_position = self._format_position(voter_responses[sample_question].answer)
            candidate_position = self._format_position(candidate_responses[sample_question].answer)

            # Create issue match detail
            issue_match = IssueMatchDetailSchema(
                issue=issue,
                alignment=alignment,
                voter_position=voter_position,
                candidate_position=candidate_position
            )
            issue_matches.append(issue_match)

        # Calculate overall match percentage (0-100) with weights
        overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0
        match_percentage = math.floor(overall_score * 100)

        # Calculate a confidence score (0.0-1.0) based on coverage
        # More questions = higher confidence
        total_questions = len(voter_submission.responses)
        questions_answered_ratio = len(common_questions) / total_questions if total_questions > 0 else 0
        confidence_score = min(1.0, 0.5 + (questions_answered_ratio * 0.5))  # Range from 0.5 to 1.0

        # Get top-aligned issues (up to 3)
        if using_explicit_priorities:
            # If using priorities, sort based on both alignment score and weight
            sorted_issues = sorted(
                [(issue, score, category_weights.get(issue, 1.0)) for issue, score in issue_scores.items()],
                key=lambda x: (x[1] * x[2], x[1]),  # Sort by weighted score, then raw score
                reverse=True
            )
            top_aligned_issues = [issue for issue, score, _ in sorted_issues[:3] if score >= 0.6]
        else:
            # Original sorting by just the score
            sorted_issues = sorted(issue_scores.items(), key=lambda x: x[1], reverse=True)
            top_aligned_issues = [issue for issue, score in sorted_issues[:3] if score >= 0.6]

        # Create list of strongest match factors for explanation
        strongest_match_factors = []
        if using_explicit_priorities:
            for issue, impact in sorted(score_contributions.items(), key=lambda x: x[1], reverse=True)[:3]:
                if impact > 0.1:  # Only include significant factors
                    priority_level = "High priority" if category_weights.get(issue, 1.0) > 0.8 else \
                        "Medium priority" if category_weights.get(issue, 1.0) > 0.4 else \
                            "Low priority"
                    alignment = self._get_alignment_level(issue_scores[issue])
                    strongest_match_factors.append(f"{priority_level} {issue} ({alignment.lower()})")

        # Convert weighted issues dictionary to schema objects
        weighted_issues_schema = [
            WeightedIssueSchema(
                category=issue["category"],
                weight=issue["weight"],
                impact_score=issue["impact_score"]
            ) for issue in weighted_issues
        ]

        # Always return the full schema, even when not using explicit priorities
        return CandidateMatchSchema(
            candidate_id=candidate.candidate_id,
            candidate_name=getattr(candidate, "name", candidate.candidate_id),
            candidate_title="Senate Candidate",  # Default title can be customized
            match_percentage=match_percentage,
            confidence_score=confidence_score,
            top_aligned_issues=top_aligned_issues,
            issue_matches=issue_matches,
            weighted_issues=weighted_issues_schema,
            strongest_match_factors=strongest_match_factors
        )

    def _extract_category_weights(self, voter_submission: VoterSubmissionSchema) -> Dict[str, float]:
        """
        Extract category weights from voter priorities or infer them if not provided.

        Returns a dictionary mapping category names to weights (0.0-1.0).
        """
        # Check if explicit priorities are provided
        if hasattr(voter_submission, 'issue_priorities') and voter_submission.issue_priorities:
            # Use explicitly provided priorities
            return self._normalize_priority_weights(voter_submission.issue_priorities)

        # No explicit priorities - use inference methods
        return self._infer_priorities(voter_submission)

    def _infer_priorities(self, voter_submission: VoterSubmissionSchema) -> Dict[str, float]:
        """
        Infer priorities when none are explicitly provided.

        Strategy options:
        1. Equal weighting (default fallback)
        2. Infer from response patterns
        3. Use population-level defaults
        """
        # Get all categories from the voter's responses
        categories = set()
        for response in voter_submission.responses:
            category = getattr(response, 'category', None)
            if category:
                categories.add(category)
            else:
                # For questions without a category, infer it
                inferred_category = self._determine_category_from_keywords(response.question)
                categories.add(inferred_category)

        # Strategy 1: Equal weights (simplest approach)
        weights = {}
        weight_value = 1.0 / len(categories) if categories else 1.0
        for category in categories:
            weights[category] = weight_value

        return weights

    @staticmethod
    def _normalize_priority_weights(priorities: List[IssuePrioritySchema]) -> Dict[str, float]:
        """
        Normalize priority weights to ensure they sum to a meaningful value.

        This handles both explicit weights and rankings.
        """
        weights = {}
        total_explicit_weight = 0

        # Check if weights are already provided
        for priority in priorities:
            weights[priority.category] = priority.weight
            total_explicit_weight += priority.weight

        # If weights seem to be rankings (e.g., 1, 2, 3) rather than weights (0.1-1.0),
        # convert them to weights inversely proportional to rank
        if any(w > 1.0 for w in weights.values()):
            # Assume these are rankings, lower is more important
            max_rank = max(weights.values())
            normalized_weights = {}
            for category, rank in weights.items():
                # Convert ranks to weights (inverse to make lower ranks higher weight)
                normalized_weights[category] = (max_rank - rank + 1) / max_rank
            return normalized_weights

        # If weights don't sum to 1.0, normalize them
        if total_explicit_weight > 0 and abs(total_explicit_weight - 1.0) > 0.01:
            for category in weights:
                weights[category] = weights[category] / total_explicit_weight

        return weights

    def _categorize_questions(self, common_questions, voter_responses, candidate_responses):
        """
        Categorize questions into issue categories based on the category field in responses.

        Args:
            common_questions: Set of common questions between voter and candidate
            voter_responses: Dictionary mapping question text to voter response

        Returns:
            Dictionary mapping categories to lists of questions
        """
        categories = {}

        for question in common_questions:
            # Get the category from voter response
            voter_response = voter_responses[question]
            category = getattr(voter_response, 'category', None)

            # If no category is provided, use a more readable name based on keywords
            if not category:
                category = self._determine_category_from_keywords(question)

            # Add the question to the appropriate category
            if category not in categories:
                categories[category] = []
            categories[category].append(question)

        return categories

    @staticmethod
    def _determine_category_from_keywords(question):
        """
        Determine category based on keywords in the question text.
        This is a fallback for when no category is provided.
        """
        keywords = {
            "language immersion": "Education Access",
            "access": "Education Access",
            "mental health": "Student Support",
            "counselors": "Student Support",
            "resources for students": "Student Support",
            "funding": "School Funding",
            "budget": "School Funding",
            "vocational": "Vocational Training",
            "career": "Vocational Training",
            "technical education": "Vocational Training",
            "sro": "School Safety",
            "officers": "School Safety",
            "safe": "School Safety",
            "community": "Community Engagement",
            "families": "Community Engagement",
            "improved": "Educational Progress",
            "progress": "Educational Progress"
        }

        # Check if any keyword is in the question
        for keyword, category in keywords.items():
            if keyword.lower() in question.lower():
                return category

        # Default category if no keywords match
        return "Other Issues"

    @staticmethod
    def _get_alignment_level(score):
        """Convert a numeric score to an alignment level string."""
        if score >= 0.8:
            return "Strongly Aligned"
        elif score >= 0.5:
            return "Moderately Aligned"
        else:
            return "Weakly Aligned"

    @staticmethod
    def _format_position(answer):
        """Format an answer into a readable position statement."""
        if isinstance(answer, bool):
            return "Yes" if answer else "No"
        elif isinstance(answer, list):
            return ", ".join(answer)
        else:
            # Limit text length for display
            text = str(answer)
            if len(text) > 100:
                return text[:97] + "..."
            return text

    @staticmethod
    def _determine_response_type(answer: Any) -> str:
        """Determine the response type based on the answer."""
        if isinstance(answer, bool):
            return 'binary'
        elif isinstance(answer, list):
            return 'multiple-choice'
        elif isinstance(answer, dict):
            return 'ranking'
        else:
            return 'text'

    @staticmethod
    def _calculate_similarity(voter_answer: Any, candidate_answer: Any, response_type: str) -> Tuple[float, str]:
        """
        Calculate similarity between voter and candidate answers.

        Returns:
            Tuple of (similarity_score, explanation)
        """
        if response_type == 'binary':
            # Direct binary comparison
            similarity = 1.0 if voter_answer == candidate_answer else 0.0
            explanation = "Exact match" if math.isclose(similarity, 1.0, rel_tol=1e-09, abs_tol=1e-09) else \
                "Different responses"

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

                if math.isclose(similarity, 1.0, rel_tol=1e-09, abs_tol=1e-09):
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

            if math.isclose(similarity, 1.0, rel_tol=1e-09, abs_tol=1e-09) and voter_top == candidate_top:
                explanation = "Exact match on priorities"
            elif similarity > 0:
                explanation = f"Partial match: {len(common_items)} common priorities"
            else:
                explanation = "Different priorities"

        else:  # text responses
            # For text responses, try to check for agreement/disagreement phrases
            voter_text = str(voter_answer).lower()
            candidate_text = str(candidate_answer).lower()

            # Check for the exact match first
            if voter_text == candidate_text:
                similarity = 1.0
                explanation = "Exact text match"
            else:
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

                    # Additional check for agreement words
                    agreement_words = {"agree", "support", "yes", "approve", "favor"}
                    disagreement_words = {"disagree", "oppose", "no", "disapprove", "against"}

                    voter_agrees = any(word in voter_text for word in agreement_words)
                    voter_disagrees = any(word in voter_text for word in disagreement_words)
                    candidate_agrees = any(word in candidate_text for word in agreement_words)
                    candidate_disagrees = any(word in candidate_text for word in disagreement_words)

                    # Boost similarity if both agree or both disagree
                    if (voter_agrees and candidate_agrees) or (voter_disagrees and candidate_disagrees):
                        similarity = max(similarity, 0.8)
                    # Reduce similarity if one agrees and one disagrees
                    elif (voter_agrees and candidate_disagrees) or (voter_disagrees and candidate_agrees):
                        similarity = min(similarity, 0.2)

                    if similarity >= 0.7:
                        explanation = "High text similarity"
                    elif similarity >= 0.3:
                        explanation = "Moderate text similarity"
                    else:
                        explanation = "Low text similarity"

        return similarity, explanation


# Create an instance for dependency injection
matching_engine = MatchingEngine()
