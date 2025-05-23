import math
import asyncio
from datetime import datetime
from typing import Any, Tuple, List, Dict, Optional

from app.utils.logging_util import setup_logger
from app.schemas.voters_schema import (
    VoterSubmissionSchema, CandidateMatchSchema, MatchResultsResponseSchema,
    IssueMatchDetailSchema, VoterValueProfileSchema, TopAlignedIssueSchema,
    QuestionTopicSchema
)
from app.services.candidate_service import candidate_service
from app.schemas.candidate_schema import CandidateResponseSchema
from app.services.llm_service import llm_service
from app.services.semantic_matching_service import semantic_service
from app.services.caching_service import cache_service
from app.core.config import settings


class EnhancedMatchingEngine:
    """
    Multi-layer hybrid matching engine that combines:
    1. Pre-processing with LLM topic discovery
    2. Real-time semantic matching with embeddings
    3. LLM validation for high-quality matches
    4. Intelligent caching for performance
    """

    def __init__(self):
        self.logger = setup_logger(__name__)

    async def process_voter_submission(self, submission: VoterSubmissionSchema) -> MatchResultsResponseSchema:
        """
        Process a voter's submission using the multi-layer hybrid approach with debugging.
        """
        self.logger.info(f"Processing submission for voter {submission.citizen_id} in election {submission.election_id}")

        # Step 1: Get candidates for the election
        candidates = await candidate_service.get_candidates_for_election(submission.election_id)
        self.logger.info(f"Retrieved {len(candidates)} candidates for election {submission.election_id}")

        if not candidates:
            self.logger.warning(f"No candidates found for election {submission.election_id}")
            return MatchResultsResponseSchema(
                citizen_id=submission.citizen_id,
                election_id=submission.election_id,
                voter_values_profile=[],
                matches=[],
                generated_at=datetime.now(),
                processing_method="no_candidates",
                confidence_score=0.0
            )

        # Step 2: Pre-process election data (discover topics if not cached)
        all_questions = self._extract_all_questions(submission, candidates)
        topics = await self._get_or_discover_topics(submission.election_id, all_questions)

        # Step 3: Generate voter profile
        voter_profile = await self._generate_voter_profile(submission, topics)

        # Step 4: Calculate matches for each candidate
        self.logger.info(f"=== CALCULATING MATCHES FOR {len(candidates)} CANDIDATES ===")

        match_tasks = [
            self._calculate_enhanced_match(submission, candidate, topics)
            for candidate in candidates
        ]

        match_results = await asyncio.gather(*match_tasks, return_exceptions=True)

        self.logger.info(f"=== MATCH RESULTS FROM ASYNCIO.GATHER ===")
        for i, result in enumerate(match_results):
            if isinstance(result, Exception):
                self.logger.error(f"Candidate {i} failed with exception: {result}")
            elif isinstance(result, CandidateMatchSchema):
                self.logger.info(f"Candidate {i}: {result.candidate_id} = {result.match_percentage}%")
            else:
                self.logger.warning(f"Candidate {i}: Unexpected result type: {type(result)}")

        # Filter out failed matches
        valid_matches = [
            match for match in match_results
            if isinstance(match, CandidateMatchSchema)
        ]

        self.logger.info(f"=== VALID MATCHES AFTER FILTERING ===")
        for match in valid_matches:
            self.logger.info(f"Valid match: {match.candidate_id} = {match.match_percentage}%")

        # Filter out zero matches
        non_zero_matches = [
            match for match in valid_matches
            if match.match_percentage > 0
        ]

        self.logger.info(f"=== NON-ZERO MATCHES AFTER FILTERING ===")
        for match in non_zero_matches:
            self.logger.info(f"Non-zero match: {match.candidate_id} = {match.match_percentage}%")

        # Step 5: Sort matches by overall score (descending)
        non_zero_matches.sort(key=lambda x: x.match_percentage, reverse=True)

        # Step 6: Determine processing method and confidence
        processing_method, confidence = self._determine_processing_quality(non_zero_matches)

        self.logger.info(f"=== FINAL RESULT ===")
        self.logger.info(f"Final matches count: {len(non_zero_matches)}")
        self.logger.info(f"Processing method: {processing_method}")
        self.logger.info(f"Confidence: {confidence}")

        return MatchResultsResponseSchema(
            citizen_id=submission.citizen_id,
            election_id=submission.election_id,
            voter_values_profile=voter_profile,
            matches=non_zero_matches,  # Use non_zero_matches instead of valid_matches
            generated_at=datetime.now(),
            processing_method=processing_method,
            confidence_score=confidence
        )

    def _extract_all_questions(self, submission: VoterSubmissionSchema,
                               candidates: List[CandidateResponseSchema]) -> List[str]:
        """Extract all unique questions from voter and candidates."""
        all_questions = set()

        # Add voter questions
        for response in submission.responses:
            all_questions.add(response.question)

        # Add candidate questions
        for candidate in candidates:
            for response in candidate.responses:
                all_questions.add(response.question)

        return list(all_questions)

    async def _get_or_discover_topics(self, election_id: str, all_questions: List[str]) -> List[QuestionTopicSchema]:
        """Get topics from cache or discover them using LLM."""
        # Try to get from cache first
        cached_topics = await cache_service.get_election_topics(election_id)

        if cached_topics and settings.ENABLE_LLM_MATCHING:
            self.logger.info(f"Using cached topics for election {election_id}")
            return [QuestionTopicSchema(**topic) for topic in cached_topics]

        # Discover topics using LLM
        if settings.ENABLE_LLM_MATCHING:
            self.logger.info(f"Discovering topics for election {election_id}")
            topics = await llm_service.discover_election_topics(
                all_questions,
                election_context=f"Election {election_id}"
            )

            # Cache the results
            if topics:
                topics_data = [topic.model_dump() for topic in topics]
                await cache_service.cache_election_topics(election_id, topics_data)

            return topics

        # Fallback: create basic topics from question categories
        return self._create_fallback_topics(all_questions)

    def _create_fallback_topics(self, all_questions: List[str]) -> List[QuestionTopicSchema]:
        """Create basic topics when LLM is not available."""
        # Group questions by basic keywords
        topic_keywords = {
            "education": ["education", "school", "student", "teacher", "learning"],
            "healthcare": ["health", "medical", "hospital", "care", "wellness"],
            "economy": ["economy", "job", "employment", "business", "financial"],
            "safety": ["safety", "security", "crime", "police", "law enforcement"],
            "environment": ["environment", "climate", "pollution", "green", "sustainability"]
        }

        topics = []
        for topic_name, keywords in topic_keywords.items():
            topic_questions = [
                q for q in all_questions
                if any(keyword.lower() in q.lower() for keyword in keywords)
            ]

            if topic_questions:
                topic = QuestionTopicSchema(
                    topic_id=topic_name,
                    topic_name=topic_name.title(),
                    topic_description=f"Questions related to {topic_name}",
                    questions=topic_questions,
                    importance_weight=1.0
                )
                topics.append(topic)

        # Add uncategorized questions
        categorized_questions = set()
        for topic in topics:
            categorized_questions.update(topic.questions)

        uncategorized = [q for q in all_questions if q not in categorized_questions]
        if uncategorized:
            topic = QuestionTopicSchema(
                topic_id="other",
                topic_name="Other Issues",
                topic_description="Other policy questions",
                questions=uncategorized,
                importance_weight=0.8
            )
            topics.append(topic)

        return topics

    async def _generate_voter_profile(self, submission: VoterSubmissionSchema,
                                      topics: List[QuestionTopicSchema]) -> List[VoterValueProfileSchema]:
        """Generate voter's political values profile."""
        voter_responses_data = [
            {
                "question": r.question,
                "answer": self._format_answer_for_analysis(r.answer),
                "category": r.category
            }
            for r in submission.responses
        ]

        # Try to get from cache first
        cached_profile = await cache_service.get_voter_profile(voter_responses_data)
        if cached_profile:
            return [VoterValueProfileSchema(**item) for item in cached_profile]

        # Generate using LLM if available
        if settings.ENABLE_LLM_MATCHING:
            try:
                profile_data = await llm_service.generate_voter_profile(voter_responses_data, topics)
                if profile_data:
                    # Cache the results
                    await cache_service.cache_voter_profile(voter_responses_data, profile_data)
                    return [VoterValueProfileSchema(**item) for item in profile_data]
            except Exception as e:
                self.logger.error(f"LLM voter profile generation failed: {str(e)}")

        # Fallback: create basic profile from responses
        return self._create_fallback_profile(submission, topics)

    def _create_fallback_profile(self, submission: VoterSubmissionSchema,
                                 topics: List[QuestionTopicSchema]) -> List[VoterValueProfileSchema]:
        """Create basic voter profile when LLM is not available."""
        profile = []

        # Group responses by category or topic
        category_responses = {}
        for response in submission.responses:
            category = response.category or "General"
            if category not in category_responses:
                category_responses[category] = []
            category_responses[category].append(response)

        # Create profile items for each category
        for category, responses in category_responses.items():
            # Determine priority based on response strength
            strong_responses = sum(1 for r in responses if self._is_strong_response(r.answer))
            priority = "High" if strong_responses > len(responses) * 0.6 else "Medium"

            # Create description
            description = f"You have expressed {priority.lower()} priority views on {category.lower()}"

            profile_item = VoterValueProfileSchema(
                issue=category,
                description=description,
                priority_level=priority
            )
            profile.append(profile_item)

        return profile

    def _is_strong_response(self, answer: Any) -> bool:
        """Determine if a response indicates strong preference."""
        if isinstance(answer, str):
            strong_words = ["strongly", "very", "extremely", "absolutely", "definitely"]
            return any(word in answer.lower() for word in strong_words)
        elif isinstance(answer, bool):
            return answer  # True indicates support
        elif isinstance(answer, list):
            return len(answer) > 2  # Multiple selections indicate engagement
        return False

    async def _calculate_enhanced_match(self, voter_submission: VoterSubmissionSchema,
                                        candidate: CandidateResponseSchema,
                                        topics: List[QuestionTopicSchema]) -> CandidateMatchSchema:
        """
        Calculate enhanced match using multi-layer approach with debug logging.
        """
        self.logger.info(f"=== CALCULATING MATCH FOR CANDIDATE {candidate.candidate_id} ===")

        voter_responses = {r.question: r for r in voter_submission.responses}
        candidate_responses = {r.question: r for r in candidate.responses}

        self.logger.info(f"Voter has {len(voter_responses)} questions, candidate has {len(candidate_responses)} questions")

        # Layer 1: Try direct question matching
        common_questions = set(voter_responses.keys()).intersection(set(candidate_responses.keys()))

        if common_questions:
            self.logger.info(f"DIRECT MATCHING: Found {len(common_questions)} common questions")
            return await self._process_direct_matches(
                voter_submission, candidate, common_questions, voter_responses, candidate_responses
            )

        # Layer 2: Try semantic matching with embeddings
        if settings.ENABLE_SEMANTIC_MATCHING and semantic_service.model:
            self.logger.info(f"SEMANTIC MATCHING: Attempting semantic matching")
            semantic_matches = await self._find_semantic_matches(voter_responses, candidate_responses)

            self.logger.info(f"SEMANTIC MATCHING: Found {len(semantic_matches)} semantic matches")

            if semantic_matches:
                # Log the actual matches found
                for voter_q, candidate_q in list(semantic_matches.items())[:3]:  # Log first 3
                    self.logger.info(f"  Match: {voter_q[:50]}... -> {candidate_q[:50]}...")

                result = await self._process_semantic_matches(
                    voter_submission, candidate, semantic_matches, voter_responses, candidate_responses
                )

                self.logger.info(f"SEMANTIC RESULT: {result.match_percentage}% match for candidate {candidate.candidate_id}")
                return result

        # Layer 3: Try LLM-based matching
        if settings.ENABLE_LLM_MATCHING:
            self.logger.info(f"LLM MATCHING: Attempting LLM matching")
            llm_matches = await self._find_llm_matches(voter_responses, candidate_responses, topics)
            if llm_matches:
                self.logger.info(f"LLM MATCHING: Found {len(llm_matches)} LLM matches")
                return await self._process_llm_matches(
                    voter_submission, candidate, llm_matches, voter_responses, candidate_responses
                )

        # No matches found
        self.logger.warning(f"NO MATCHES: No matches found for candidate {candidate.candidate_id}")
        return CandidateMatchSchema(
            candidate_id=candidate.candidate_id,
            candidate_name=getattr(candidate, "name", candidate.candidate_id),
            candidate_title="Candidate",
            candidate_image_url=getattr(candidate, "image_url", None),
            match_percentage=0,
            match_strength_visual=0.0,
            top_aligned_issues=[],
            issue_matches=[],
            overall_explanation="No comparable questions found between voter and candidate responses."
        )

    async def _find_semantic_matches(self, voter_responses: Dict,
                                     candidate_responses: Dict) -> Dict[str, str]:
        """Find semantic matches between voter and candidate questions."""
        voter_questions = list(voter_responses.keys())
        candidate_questions = list(candidate_responses.keys())

        self.logger.debug(f"Looking for semantic matches between {len(voter_questions)} voter questions and {len(candidate_questions)} candidate questions")

        # Use batch processing for efficiency
        batch_results = semantic_service.batch_find_similar_questions(
            voter_questions, candidate_questions
        )

        # Convert to simple mapping (voter_q -> best_candidate_q)
        matches = {}
        for voter_q, similarities in batch_results.items():
            if similarities:
                best_match = similarities[0]  # Already sorted by score
                self.logger.debug(f"Semantic match found: {voter_q[:50]}... -> {best_match.candidate_question[:50]}... (score: {best_match.similarity_score:.3f})")
                matches[voter_q] = best_match.candidate_question
            else:
                self.logger.debug(f"No semantic match found for: {voter_q[:50]}...")

        self.logger.info(f"Found {len(matches)} semantic question matches")
        return matches

    async def _find_llm_matches(self, voter_responses: Dict, candidate_responses: Dict,
                                topics: List[QuestionTopicSchema]) -> Dict[str, str]:
        """Find LLM-based matches between questions."""
        matches = {}

        for voter_q in voter_responses.keys():
            # Try to get from cache first
            candidate_questions = list(candidate_responses.keys())
            cached_similarities = await cache_service.get_question_similarities(voter_q, candidate_questions)

            if cached_similarities:
                # Use cached result
                if cached_similarities:
                    best_match = max(cached_similarities, key=lambda x: x.get("similarity_score", 0))
                    if best_match.get("similarity_score", 0) >= 0.6:
                        matches[voter_q] = best_match["candidate_question"]
            else:
                # Use LLM to find matches
                try:
                    # Find relevant topic context
                    topic_context = ""
                    for topic in topics:
                        if voter_q in topic.questions:
                            topic_context = topic.topic_description
                            break

                    similarities = await llm_service.find_similar_questions(
                        voter_q, candidate_questions, topic_context
                    )

                    if similarities:
                        best_similarity = similarities[0]
                        if best_similarity.similarity_score >= 0.6:
                            matches[voter_q] = best_similarity.candidate_question

                        # Cache the results
                        similarities_data = [sim.model_dump() for sim in similarities]
                        await cache_service.cache_question_similarities(voter_q, candidate_questions, similarities_data)

                except Exception as e:
                    self.logger.error(f"LLM question matching failed: {str(e)}")

        return matches

    async def _process_direct_matches(self, voter_submission, candidate, common_questions,
                                      voter_responses, candidate_responses) -> CandidateMatchSchema:
        """Process direct question matches."""
        issue_matches = []
        issue_scores = {}

        # Group questions by category
        categories = {}
        for question in common_questions:
            voter_response = voter_responses[question]
            category = voter_response.category or self._determine_category_from_keywords(question)

            if category not in categories:
                categories[category] = []
            categories[category].append(question)

        # Calculate scores for each category
        for category, questions in categories.items():
            scores = []

            for question in questions:
                voter_answer = voter_responses[question].answer
                candidate_answer = candidate_responses[question].answer

                similarity, explanation = await self._calculate_enhanced_similarity(
                    voter_answer, candidate_answer, question
                )
                scores.append(similarity)

            # Calculate category score
            avg_score = sum(scores) / len(scores) if scores else 0
            issue_scores[category] = avg_score

            # Create issue match detail
            sample_question = questions[0]
            voter_position = self._format_position(voter_responses[sample_question].answer)
            candidate_position = self._format_position(candidate_responses[sample_question].answer)

            issue_match = IssueMatchDetailSchema(
                issue=category,
                alignment=self._get_alignment_level(avg_score),
                alignment_score=avg_score,
                voter_position=voter_position,
                candidate_position=candidate_position,
                explanation=f"Based on {len(questions)} question(s) in this area"
            )
            issue_matches.append(issue_match)

        return self._build_match_result(candidate, issue_scores, issue_matches, "direct")

    async def _process_semantic_matches(self, voter_submission, candidate, semantic_matches,
                                        voter_responses, candidate_responses) -> CandidateMatchSchema:
        """Process semantic matches with comprehensive error handling."""

        self.logger.info(f"Processing {len(semantic_matches)} semantic matches for candidate {candidate.candidate_id}")

        try:
            if not semantic_matches:
                return self._create_no_match_result(candidate)

            # Group matches by category first
            category_matches = {}

            for voter_q, candidate_q in semantic_matches.items():
                try:
                    voter_response = voter_responses.get(voter_q)
                    candidate_response = candidate_responses.get(candidate_q)

                    if not voter_response or not candidate_response:
                        self.logger.warning(f"Missing response data for match: {voter_q} -> {candidate_q}")
                        continue

                    # Get category
                    category = getattr(voter_response, 'category', None) or self._determine_category_from_keywords(voter_q)

                    if category not in category_matches:
                        category_matches[category] = []

                    category_matches[category].append({
                        'voter_q': voter_q,
                        'candidate_q': candidate_q,
                        'voter_answer': voter_response.answer,
                        'candidate_answer': candidate_response.answer
                    })

                except Exception as e:
                    self.logger.error(f"Error grouping semantic match {voter_q} -> {candidate_q}: {e}")
                    continue

            if not category_matches:
                self.logger.warning(f"No valid category matches created for candidate {candidate.candidate_id}")
                return self._create_no_match_result(candidate)

            # Process each category
            issue_matches = []
            issue_scores = {}

            for category, matches in category_matches.items():
                try:
                    # Calculate scores for all questions in this category
                    scores = []

                    for match in matches:
                        try:
                            similarity = self._calculate_smart_similarity(match['voter_answer'], match['candidate_answer'])
                            scores.append(similarity)
                            self.logger.debug(f"Category {category}: {similarity:.3f} similarity for {match['voter_answer']} vs {match['candidate_answer']}")
                        except Exception as e:
                            self.logger.error(f"Error calculating similarity in {category}: {e}")
                            scores.append(0.5)  # Default score if calculation fails

                    if not scores:
                        self.logger.warning(f"No scores calculated for category {category}")
                        continue

                    # Average score for this category
                    avg_score = sum(scores) / len(scores)
                    issue_scores[category] = avg_score

                    self.logger.info(f"Category {category}: {avg_score:.3f} average score from {len(scores)} questions")

                    # Create issue match detail using the first match as representative
                    first_match = matches[0]
                    issue_match = IssueMatchDetailSchema(
                        issue=category,
                        alignment=self._get_alignment_level(avg_score),
                        alignment_score=avg_score,
                        voter_position=self._format_position(first_match['voter_answer']),
                        candidate_position=self._format_position(first_match['candidate_answer']),
                        explanation=f"Based on {len(matches)} semantic question match(es) with {avg_score:.0%} alignment"
                    )
                    issue_matches.append(issue_match)

                except Exception as e:
                    self.logger.error(f"Error processing category {category}: {e}")
                    continue

            if not issue_matches:
                self.logger.warning(f"No valid issue matches created for candidate {candidate.candidate_id}")
                return self._create_no_match_result(candidate)

            # Calculate overall match - ensure we have valid scores
            valid_scores = [score for score in issue_scores.values() if score > 0]

            if not valid_scores:
                self.logger.warning(f"No valid scores > 0 for candidate {candidate.candidate_id}")
                return self._create_no_match_result(candidate)

            overall_score = sum(valid_scores) / len(valid_scores)
            match_percentage = max(1, min(100, int(overall_score * 100)))  # Ensure 1-100 range

            self.logger.info(f"Candidate {candidate.candidate_id}: overall_score={overall_score:.3f}, match_percentage={match_percentage}")

            # Create top aligned issues
            colors = ["#FF6B35", "#F7931E", "#FFD23F", "#06FFA5", "#118AB2"]
            top_aligned_issues = []

            # Sort issues by score and take top 5
            sorted_issues = sorted(issue_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (issue, score) in enumerate(sorted_issues[:5]):
                if score >= 0.3:  # Lower threshold to ensure we get some results
                    top_aligned_issues.append(TopAlignedIssueSchema(
                        issue=issue,
                        color=colors[i % len(colors)]
                    ))

            result = CandidateMatchSchema(
                candidate_id=candidate.candidate_id,
                candidate_name=getattr(candidate, "name", None) or candidate.candidate_id,
                candidate_title=getattr(candidate, "title", None) or "Candidate",
                candidate_image_url=getattr(candidate, "image_url", None),
                match_percentage=match_percentage,
                match_strength_visual=overall_score,
                top_aligned_issues=top_aligned_issues,
                issue_matches=issue_matches,
                overall_explanation=f"Strong semantic alignment found across {len(semantic_matches)} policy questions with {overall_score:.0%} average agreement."
            )

            self.logger.info(f"SUCCESS: Created match result with {match_percentage}% for candidate {candidate.candidate_id}")
            return result

        except Exception as e:
            self.logger.error(f"CRITICAL ERROR in _process_semantic_matches for candidate {candidate.candidate_id}: {e}")
            # Return a fallback result instead of raising
            return self._create_no_match_result(candidate)

    def _calculate_smart_similarity(self, voter_answer: Any, candidate_answer: Any) -> float:
        """Fixed smart similarity calculation."""

        self.logger.debug(f"Calculating similarity: voter='{voter_answer}' ({type(voter_answer)}) vs candidate='{candidate_answer}' ({type(candidate_answer)})")

        # Convert everything to strings for comparison
        voter_str = str(voter_answer).lower().strip()
        candidate_str = str(candidate_answer).lower().strip()

        # Handle exact matches
        if voter_str == candidate_str:
            self.logger.debug("Exact match found: 1.0")
            return 1.0

        # More precise boolean-like response detection
        voter_is_positive = any(word in voter_str for word in ["strongly agree", "agree", "yes", "support", "favor"])
        voter_is_negative = any(word in voter_str for word in ["strongly disagree", "disagree", "no", "oppose", "against"])

        candidate_is_true = candidate_answer is True
        candidate_is_false = candidate_answer is False

        # Debug the detection
        self.logger.debug(f"Voter positive: {voter_is_positive}, negative: {voter_is_negative}")
        self.logger.debug(f"Candidate true: {candidate_is_true}, false: {candidate_is_false}")

        # Positive voter + True candidate = high similarity
        if voter_is_positive and candidate_is_true:
            self.logger.debug("Positive voter + True candidate: 0.9")
            return 0.9

        # Negative voter + False candidate = high similarity
        if voter_is_negative and candidate_is_false:
            self.logger.debug("Negative voter + False candidate: 0.9")
            return 0.9

        # Positive voter + False candidate = low similarity
        if voter_is_positive and candidate_is_false:
            self.logger.debug("Positive voter + False candidate: 0.1")
            return 0.1

        # Negative voter + True candidate = low similarity
        if voter_is_negative and candidate_is_true:
            self.logger.debug("Negative voter + True candidate: 0.1")
            return 0.1

        # Both are actual booleans
        if isinstance(voter_answer, bool) and isinstance(candidate_answer, bool):
            similarity = 1.0 if voter_answer == candidate_answer else 0.0
            self.logger.debug(f"Both boolean: {similarity}")
            return similarity

        # Default case - they were found semantically similar, so give decent score
        self.logger.debug("Default semantic match: 0.7")
        return 0.7

    async def _process_llm_matches(self, voter_submission, candidate, llm_matches,
                                   voter_responses, candidate_responses) -> CandidateMatchSchema:
        """Process LLM matches."""
        return await self._process_question_matches(
            candidate, llm_matches, voter_responses, candidate_responses, "llm_enhanced"
        )

    async def _process_question_matches(self, candidate, question_matches, voter_responses,
                                        candidate_responses, method) -> CandidateMatchSchema:
        """Generic method to process question matches with better error handling."""
        self.logger.debug(f"Processing {len(question_matches)} question matches for candidate {candidate.candidate_id} using {method}")

        if not question_matches:
            self.logger.warning(f"No question matches provided for candidate {candidate.candidate_id}")
            return self._create_no_match_result(candidate)

        issue_matches = []
        issue_scores = {}

        # Group matches by category
        categories = {}
        for voter_q, candidate_q in question_matches.items():
            voter_response = voter_responses.get(voter_q)
            if not voter_response:
                self.logger.warning(f"Voter response not found for question: {voter_q}")
                continue

            candidate_response = candidate_responses.get(candidate_q)
            if not candidate_response:
                self.logger.warning(f"Candidate response not found for question: {candidate_q}")
                continue

            category = getattr(voter_response, 'category', None) or self._determine_category_from_keywords(voter_q)

            if category not in categories:
                categories[category] = []
            categories[category].append((voter_q, candidate_q))

        self.logger.debug(f"Grouped questions into {len(categories)} categories: {list(categories.keys())}")

        # Calculate scores for each category
        for category, question_pairs in categories.items():
            scores = []

            for voter_q, candidate_q in question_pairs:
                try:
                    voter_answer = voter_responses[voter_q].answer
                    candidate_answer = candidate_responses[candidate_q].answer

                    self.logger.debug(f"Comparing answers for {category}: voter='{voter_answer}' vs candidate='{candidate_answer}'")

                    similarity, explanation = await self._calculate_enhanced_similarity(
                        voter_answer, candidate_answer, f"{voter_q} | {candidate_q}"
                    )
                    scores.append(similarity)

                    self.logger.debug(f"Similarity score for {category}: {similarity:.3f} - {explanation}")

                except Exception as e:
                    self.logger.error(f"Error calculating similarity for {category}: {str(e)}")
                    scores.append(0.0)

            if not scores:
                self.logger.warning(f"No valid scores calculated for category {category}")
                continue

            # Calculate category score
            avg_score = sum(scores) / len(scores)
            issue_scores[category] = avg_score

            self.logger.debug(f"Category {category} average score: {avg_score:.3f}")

            # Create issue match detail
            sample_voter_q, sample_candidate_q = question_pairs[0]
            voter_position = self._format_position(voter_responses[sample_voter_q].answer)
            candidate_position = self._format_position(candidate_responses[sample_candidate_q].answer)

            issue_match = IssueMatchDetailSchema(
                issue=category,
                alignment=self._get_alignment_level(avg_score),
                alignment_score=avg_score,
                voter_position=voter_position,
                candidate_position=candidate_position,
                explanation=f"Based on {len(question_pairs)} similar question(s) using {method} matching"
            )
            issue_matches.append(issue_match)

        if not issue_scores:
            self.logger.warning(f"No valid issue scores calculated for candidate {candidate.candidate_id}")
            return self._create_no_match_result(candidate)

        return self._build_match_result(candidate, issue_scores, issue_matches, method)

    def _create_no_match_result(self, candidate) -> CandidateMatchSchema:
        """Create a result for candidates with no matches."""
        return CandidateMatchSchema(
            candidate_id=candidate.candidate_id,
            candidate_name=getattr(candidate, "name", candidate.candidate_id),
            candidate_title=getattr(candidate, "title", "Candidate"),
            candidate_image_url=getattr(candidate, "image_url", None),
            match_percentage=0,
            match_strength_visual=0.0,
            top_aligned_issues=[],
            issue_matches=[],
            overall_explanation="No comparable questions found between voter and candidate responses."
        )

    async def _calculate_enhanced_similarity(self, voter_answer: Any, candidate_answer: Any,
                                             question_context: str) -> Tuple[float, str]:
        """Calculate similarity with enhanced logging and error handling."""
        self.logger.debug(f"Calculating similarity for: voter='{voter_answer}' vs candidate='{candidate_answer}'")

        # Try LLM-based analysis first (if enabled and available)
        if settings.ENABLE_LLM_MATCHING:
            try:
                # Check cache first
                cached_alignment = await cache_service.get_position_alignment(
                    str(voter_answer), str(candidate_answer), question_context
                )

                if cached_alignment:
                    self.logger.debug("Using cached LLM alignment result")
                    return cached_alignment["alignment_score"], cached_alignment["explanation"]

                # Use LLM for analysis
                score, explanation = await llm_service.analyze_position_alignment(
                    str(voter_answer), str(candidate_answer), question_context
                )

                # Cache the result
                alignment_data = {"alignment_score": score, "explanation": explanation}
                await cache_service.cache_position_alignment(
                    str(voter_answer), str(candidate_answer), question_context, alignment_data
                )

                self.logger.debug(f"LLM similarity result: {score:.3f} - {explanation}")
                return score, explanation

            except Exception as e:
                self.logger.error(f"LLM similarity analysis failed: {str(e)}")

        # Fallback to basic similarity calculation
        score, explanation = self._calculate_basic_similarity(voter_answer, candidate_answer)
        self.logger.debug(f"Basic similarity result: {score:.3f} - {explanation}")
        return score, explanation

    def _calculate_basic_similarity(self, voter_answer: Any, candidate_answer: Any) -> Tuple[float, str]:
        """Improved basic similarity calculation with better logging."""
        response_type = self._determine_response_type(voter_answer)
        self.logger.debug(f"Response type detected: {response_type}")

        if response_type == 'binary':
            similarity = 1.0 if voter_answer == candidate_answer else 0.0
            explanation = "Exact match" if similarity == 1.0 else "Different responses"

        elif response_type == 'multiple-choice':
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

        elif response_type == 'text':
            voter_text = str(voter_answer).lower().strip()
            candidate_text = str(candidate_answer).lower().strip()

            if voter_text == candidate_text:
                similarity = 1.0
                explanation = "Exact text match"
            else:
                similarity, explanation = self._analyze_text_agreement(voter_text, candidate_text)

        else:  # ranking or other
            similarity = 0.5
            explanation = "Partial similarity (complex response type)"

        self.logger.debug(f"Basic similarity calculated: {similarity:.3f} - {explanation}")
        return similarity, explanation

    def _analyze_text_agreement(self, voter_text: str, candidate_text: str) -> Tuple[float, str]:
        """Analyze text agreement with improved logic."""

        # Handle boolean-like text responses
        voter_bool = self._text_to_boolean(voter_text)
        candidate_bool = self._text_to_boolean(candidate_text)

        self.logger.debug(f"Boolean conversion: voter={voter_bool}, candidate={candidate_bool}")

        if voter_bool is not None and candidate_bool is not None:
            if voter_bool == candidate_bool:
                similarity = 0.9  # High similarity for matching boolean intent
                explanation = "Both have the same stance (support/oppose)"
            else:
                similarity = 0.1  # Low similarity for opposing stances
                explanation = "Opposing stances (one supports, one opposes)"
            return similarity, explanation

        # Handle agree/disagree patterns
        voter_agreement = self._detect_agreement_level(voter_text)
        candidate_agreement = self._detect_agreement_level(candidate_text)

        self.logger.debug(f"Agreement levels: voter={voter_agreement}, candidate={candidate_agreement}")

        if voter_agreement is not None and candidate_agreement is not None:
            # Both express clear agreement levels
            agreement_diff = abs(voter_agreement - candidate_agreement)

            if agreement_diff <= 0.2:  # Very close agreement levels
                similarity = 0.9
                explanation = "Very similar levels of agreement"
            elif agreement_diff <= 0.4:  # Somewhat close
                similarity = 0.7
                explanation = "Moderately similar positions"
            elif agreement_diff <= 0.6:  # Different but not opposite
                similarity = 0.4
                explanation = "Somewhat different positions"
            else:  # Very different or opposite
                similarity = 0.1
                explanation = "Very different positions"

            return similarity, explanation

        # Fallback to word similarity
        voter_words = set(voter_text.split())
        candidate_words = set(candidate_text.split())

        if not voter_words or not candidate_words:
            return 0.0, "One or both responses are empty"

        common_words = voter_words.intersection(candidate_words)
        all_words = voter_words.union(candidate_words)
        word_similarity = len(common_words) / len(all_words)

        if word_similarity >= 0.7:
            return word_similarity, "High text similarity"
        elif word_similarity >= 0.3:
            return word_similarity, "Moderate text similarity"
        else:
            return word_similarity, "Low text similarity"

    def _text_to_boolean(self, text: str) -> Optional[bool]:
        """Convert text responses to boolean intent."""
        text = text.lower().strip()

        # Strong positive indicators
        if any(phrase in text for phrase in ["strongly agree", "strongly support", "yes", "true", "definitely"]):
            return True

        # Strong negative indicators
        if any(phrase in text for phrase in ["strongly disagree", "strongly oppose", "no", "false", "definitely not"]):
            return False

        # Moderate positive
        if any(phrase in text for phrase in ["agree", "support", "favor", "approve"]):
            return True

        # Moderate negative
        if any(phrase in text for phrase in ["disagree", "oppose", "against", "disapprove"]):
            return False

        return None

    def _detect_agreement_level(self, text: str) -> Optional[float]:
        """Detect the level of agreement in text (0.0 = strong disagree, 1.0 = strong agree)."""
        text = text.lower().strip()

        # Strong agreement
        if any(phrase in text for phrase in ["strongly agree", "completely agree", "absolutely"]):
            return 1.0

        # Strong disagreement
        if any(phrase in text for phrase in ["strongly disagree", "completely disagree", "absolutely not"]):
            return 0.0

        # Moderate agreement
        if any(phrase in text for phrase in ["agree", "support", "yes", "favor"]):
            return 0.7

        # Moderate disagreement
        if any(phrase in text for phrase in ["disagree", "oppose", "no", "against"]):
            return 0.3

        # Neutral/unsure
        if any(phrase in text for phrase in ["neutral", "unsure", "maybe", "somewhat"]):
            return 0.5

        return None

    def _build_match_result(self, candidate, issue_scores, issue_matches, method) -> CandidateMatchSchema:
        """Build the final match result."""
        # Calculate overall match percentage
        overall_score = sum(issue_scores.values()) / len(issue_scores) if issue_scores else 0
        match_percentage = math.floor(overall_score * 100)

        # Get top-aligned issues with colors
        sorted_issues = sorted(issue_scores.items(), key=lambda x: x[1], reverse=True)
        top_aligned_issues = []

        colors = ["#FF6B35", "#F7931E", "#FFD23F", "#06FFA5", "#118AB2"]  # Orange, yellow, green, blue palette

        for i, (issue, score) in enumerate(sorted_issues[:5]):
            if score >= 0.6:  # Only include reasonably aligned issues
                color = colors[i % len(colors)]
                top_issue = TopAlignedIssueSchema(
                    issue=issue,
                    color=color
                )
                top_aligned_issues.append(top_issue)

        # Generate overall explanation if using advanced methods
        overall_explanation = None
        if method in ["llm_enhanced", "semantic"]:
            if match_percentage >= 80:
                overall_explanation = "Strong alignment across multiple policy areas with similar priorities and approaches."
            elif match_percentage >= 60:
                overall_explanation = "Good alignment on key issues with some differences in approach or emphasis."
            elif match_percentage >= 40:
                overall_explanation = "Moderate alignment with agreement on some issues but differences on others."
            else:
                overall_explanation = "Limited alignment with significant differences in policy positions."

        return CandidateMatchSchema(
            candidate_id=candidate.candidate_id,
            candidate_name=getattr(candidate, "name", candidate.candidate_id),
            candidate_title=getattr(candidate, "title", "Candidate"),
            candidate_image_url=getattr(candidate, "image_url", None),
            match_percentage=match_percentage,
            match_strength_visual=overall_score,
            top_aligned_issues=top_aligned_issues,
            issue_matches=issue_matches,
            overall_explanation=overall_explanation
        )

    def _determine_processing_quality(self, matches: List[CandidateMatchSchema]) -> Tuple[str, float]:
        """Determine the processing method used and confidence score."""
        if not matches:
            return "no_matches", 0.0

        # Check if we have any high-quality matches
        high_quality = any(match.match_percentage >= 70 for match in matches)
        has_explanations = any(match.overall_explanation for match in matches)

        if has_explanations and high_quality:
            return "llm_enhanced", 0.9
        elif high_quality:
            return "semantic", 0.8
        elif matches:
            return "basic", 0.6
        else:
            return "fallback", 0.3

    # Utility methods (keeping existing ones)

    @staticmethod
    def _determine_category_from_keywords(question):
        """Determine category based on keywords in the question text."""
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

        for keyword, category in keywords.items():
            if keyword.lower() in question.lower():
                return category

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
            return ", ".join(str(item) for item in answer)
        else:
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

    def _format_answer_for_analysis(self, answer: Any) -> str:
        """Format answer for LLM analysis."""
        if isinstance(answer, bool):
            return "Yes" if answer else "No"
        elif isinstance(answer, list):
            return ", ".join(str(item) for item in answer)
        elif isinstance(answer, dict):
            return str(answer)
        else:
            return str(answer)


# Create an instance for dependency injection
enhanced_matching_engine = EnhancedMatchingEngine()