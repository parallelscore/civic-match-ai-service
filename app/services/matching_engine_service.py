# app/services/matching_engine_service.py

import hashlib
from typing import List, Tuple
from datetime import datetime

from app.utils.logging_util import setup_logger
from app.core.matching_config import matching_config
from app.services.caching_service import cache_service
from app.services.candidate_service import candidate_service
from app.services.position_inference_service import position_inference_service
from app.services.consistency_analyzer_service import consistency_analyzer_service
from app.services.policy_dimension_discovery_service import policy_dimension_discovery_service
from app.services.enhanced_matching_calculator_service import enhanced_matching_calculator_service
from app.schemas.policy_matching_schema import PersonPolicyProfile, PolicyPosition, EnhancedMatchResult
from app.schemas.voters_schema import VoterSubmissionSchema, MatchResultsResponseSchema, VoterValueProfileSchema


class MatchingEngineService:
    """
    Main enhanced matching engine that orchestrates the multi-layer matching process
    """

    def __init__(self):
        self.logger = setup_logger(__name__)

    async def process_voter_submission(self, submission: VoterSubmissionSchema) -> MatchResultsResponseSchema:
        """
        Process voter submission using the enhanced policy-based matching system.
        Returns ALL matched candidates (not limited to top 3).

        Pipeline:
          1. Validate voter has enough policy-relevant responses (quality gate)
          2. Fetch candidates and separate eligible vs ineligible
          3. Discover policy dimensions for this election
          4. Build voter + candidate policy profiles
          5. Score every eligible candidate
          6. Assign TOP / OTHER / UNMATCH categories
          7. Generate voter values profile
        """
        self.logger.info(
            f"Processing submission for voter {submission.citizen_id} "
            f"in election {submission.election_id}"
        )

        try:
            # ── Step 1: Voter quality gate ──────────────────────────────
            quality_issue = self._check_voter_response_quality(submission)
            if quality_issue:
                self.logger.warning(
                    f"Voter {submission.citizen_id} failed quality gate: {quality_issue}"
                )
                return self._create_insufficient_responses_response(submission, quality_issue)

            # ── Step 2: Fetch & partition candidates ────────────────────
            candidates = await candidate_service.get_candidates_for_election(submission.election_id)
            if not candidates:
                self.logger.warning(f"No candidates found for election {submission.election_id}")
                return self._create_no_candidates_response(submission)

            # With lenient eligibility: eligible = has responses, ineligible = no responses
            eligible_candidates = [c for c in candidates if c.is_eligible_for_matching()]
            ineligible_candidates = [c for c in candidates if not c.is_eligible_for_matching()]

            self.logger.info(
                f"Candidates: {len(candidates)} total, "
                f"{len(eligible_candidates)} with responses (matchable), "
                f"{len(ineligible_candidates)} without responses"
            )

            # ── Step 3: Discover policy dimensions ──────────────────────
            all_questions = self._extract_all_questions(submission, eligible_candidates)
            election_analysis = await policy_dimension_discovery_service.discover_election_policy_dimensions(
                submission.election_id, all_questions
            )
            self.logger.info(
                f"Discovered {len(election_analysis.discovered_dimensions)} policy dimensions"
            )

            # ── Step 4: Build voter policy profile ──────────────────────
            voter_profile = await self._create_voter_policy_profile(submission, election_analysis)

            # ── Step 5: Score eligible candidates ───────────────────────
            enhanced_matches = []

            if eligible_candidates:
                candidate_profiles = await self._create_candidate_policy_profiles(
                    eligible_candidates, election_analysis
                )
                
                # Log dimension coverage comparison for debugging zero overlap issues
                voter_dimensions_after_filter = sorted(set(
                    pos.dimension_id for pos in voter_profile.policy_positions 
                    if pos.confidence >= matching_config.min_confidence_threshold
                ))
                self.logger.info(
                    f"Starting matching with voter dimensions (after confidence filter): {voter_dimensions_after_filter}"
                )
                
                for candidate_profile in candidate_profiles:
                    candidate_dimensions_after_filter = sorted(set(
                        pos.dimension_id for pos in candidate_profile.policy_positions 
                        if pos.confidence >= matching_config.min_confidence_threshold
                    ))
                    common_dims = sorted(set(voter_dimensions_after_filter) & set(candidate_dimensions_after_filter))
                    
                    # Log potential zero overlap issues BEFORE matching
                    if len(common_dims) < matching_config.min_dimension_overlap:
                        self.logger.warning(
                            f"PRE-MATCH WARNING: Candidate {candidate_profile.person_id} has only "
                            f"{len(common_dims)} common dimensions with voter (minimum {matching_config.min_dimension_overlap}). "
                            f"Voter dimensions: {voter_dimensions_after_filter}, "
                            f"Candidate dimensions: {candidate_dimensions_after_filter}, "
                            f"Common: {common_dims}. This will result in 0% match."
                        )
                    
                    match_result = await enhanced_matching_calculator_service.calculate_enhanced_match(
                        voter_profile, candidate_profile
                    )
                    enhanced_matches.append(self._convert_to_output_format(match_result))

            # Candidates without responses get 0%
            for candidate in ineligible_candidates:
                enhanced_matches.append(self._create_zero_match_result(candidate))
                self.logger.debug(f"0% match assigned to candidate {candidate.candidate_id} (no responses)")

            # ── Step 6: Sort and categorise ─────────────────────────────
            enhanced_matches.sort(key=lambda x: x.match_percentage, reverse=True)
            self._assign_match_categories(enhanced_matches)

            self.logger.info(
                f"Returning {len(enhanced_matches)} matches "
                f"({len(eligible_candidates)} attempted matching, {len(ineligible_candidates)} without responses)"
            )

            # ── Step 7: Voter values profile & quality metadata ─────────
            voter_values_profile = await self._generate_voter_values_profile(
                voter_profile, election_analysis
            )
            processing_method, confidence = self._determine_processing_quality(
                enhanced_matches, election_analysis, len(eligible_candidates), len(ineligible_candidates)
            )

            self.logger.info(f"Processing method: {processing_method}, Confidence: {confidence:.2f}")

            return MatchResultsResponseSchema(
                citizen_id=submission.citizen_id,
                election_id=submission.election_id,
                voter_values_profile=voter_values_profile,
                matches=enhanced_matches,
                generated_at=datetime.now(),
                processing_method=processing_method,
                confidence_score=confidence,
            )

        except Exception as e:
            self.logger.error(f"Matching pipeline failed: {str(e)}")
            return self._create_error_response(submission, str(e))

    @staticmethod
    def _check_voter_response_quality(submission: VoterSubmissionSchema) -> str:
        """
        Quality gate: ensure the voter has provided enough policy-relevant content
        before we run the (expensive) matching pipeline.

        Returns an empty string if quality is acceptable, or a human-readable
        reason string if the submission should be rejected.
        """
        # Personal-info keywords that indicate a response is NOT policy-related
        personal_info_keywords = {
            "name", "age", "gender", "email", "phone", "address",
            "occupation", "city", "state", "zip", "dob", "birthday",
            "first name", "last name", "full name", "date of birth",
        }

        policy_response_count = 0

        for response in submission.responses:
            question_lower = response.question.lower().strip()

            # Skip questions that are clearly asking for personal information
            if any(kw in question_lower for kw in personal_info_keywords):
                continue

            answer = response.answer

            # Boolean answers on non-personal questions always count
            if isinstance(answer, bool):
                policy_response_count += 1
                continue

            # List / dict answers (multi-select) count directly
            if isinstance(answer, (list, dict)):
                policy_response_count += 1
                continue

            # Text answers: structured short answers (yes/no/agree etc.) always
            # count because they carry clear directional intent.
            # Free-text answers must meet the minimum length threshold.
            if isinstance(answer, str):
                stripped = answer.strip().lower()
                structured_tokens = {
                    "yes", "no", "true", "false",
                    "agree", "disagree",
                    "support", "oppose",
                    "favor", "against",
                    "strongly agree", "strongly disagree",
                    "strongly support", "strongly oppose",
                    "strongly favor", "strongly against",
                    "neutral", "unsure", "maybe", "somewhat",
                }
                if stripped in structured_tokens:
                    # Short structured answer — counts regardless of length
                    policy_response_count += 1
                elif len(stripped) >= matching_config.min_answer_length_for_text and not stripped.isdigit():
                    # Free-text — must be long enough and not purely numeric
                    policy_response_count += 1

        if policy_response_count < matching_config.min_policy_responses:
            return (
                f"Only {policy_response_count} policy-relevant response(s) detected. "
                f"At least {matching_config.min_policy_responses} are required to generate "
                f"a meaningful match. Please answer more policy questions."
            )

        return ""  # Quality check passed

    @staticmethod
    def _extract_all_questions(submission: VoterSubmissionSchema, candidates: List) -> List[str]:
        """Extract all unique questions from voter and candidates"""

        all_questions = set()

        # Add voter questions
        for response in submission.responses:
            all_questions.add(response.question)

        # Add candidate questions
        for candidate in candidates:
            for response in candidate.responses:
                all_questions.add(response.question)

        return list(all_questions)

    async def _create_voter_policy_profile(self, submission: VoterSubmissionSchema, election_analysis) -> PersonPolicyProfile:
        """Create comprehensive policy profile for voter"""

        policy_positions = []

        # Create mapping lookup for efficiency
        question_to_dimension = {}
        for mapping in election_analysis.question_mappings:
            question_to_dimension[mapping.question] = mapping

        # Process each voter response
        for response in submission.responses:
            question = response.question

            if question in question_to_dimension:
                mapping = question_to_dimension[question]

                # Find the corresponding dimension
                dimension = next(
                    (d for d in election_analysis.discovered_dimensions if d.dimension_id == mapping.primary_dimension_id),
                    None
                )

                if dimension:
                    # Infer position for primary dimension
                    # Use comment if the voter provided one; empty string otherwise
                    voter_comment = getattr(response, 'comment', '') or ''
                    position = await position_inference_service.infer_policy_position(
                        question=question,
                        answer=response.answer,
                        comment=voter_comment,
                        dimension=dimension,
                        person_type="voter"
                    )
                    # None means the answer was not policy-relevant — skip it
                    if position is not None:
                        policy_positions.append(position)

                    # Handle secondary dimensions if they exist
                    for sec_dim_id in mapping.secondary_dimension_ids:
                        sec_dimension = next(
                            (d for d in election_analysis.discovered_dimensions if d.dimension_id == sec_dim_id),
                            None
                        )
                        if sec_dimension:
                            sec_weight = mapping.secondary_weights.get(
                                sec_dim_id, matching_config.position_basic_weight
                            )
                            sec_position = await position_inference_service.infer_policy_position(
                                question=question,
                                answer=response.answer,
                                comment=voter_comment,
                                dimension=sec_dimension,
                                person_type="voter"
                            )
                            if sec_position is not None:
                                sec_position.confidence *= sec_weight
                                sec_position.intensity_multiplier *= sec_weight
                                policy_positions.append(sec_position)

        # Analyze consistency
        tensions, consistency_score = consistency_analyzer_service.analyze_position_consistency(policy_positions)

        # Log voter dimension coverage for debugging zero overlap issues
        voter_dimension_ids = sorted(set(pos.dimension_id for pos in policy_positions))
        self.logger.info(
            f"Voter profile created with {len(policy_positions)} positions across "
            f"{len(voter_dimension_ids)} unique dimensions: {voter_dimension_ids}"
        )
        
        # Log any low-confidence positions that might be filtered out later
        low_confidence_positions = [
            pos for pos in policy_positions 
            if pos.confidence < matching_config.min_confidence_threshold
        ]
        if low_confidence_positions:
            self.logger.warning(
                f"Voter has {len(low_confidence_positions)} positions below confidence threshold "
                f"({matching_config.min_confidence_threshold}): "
                f"{[pos.dimension_id for pos in low_confidence_positions]}"
            )

        return PersonPolicyProfile(
            person_id=submission.citizen_id,
            person_type="voter",
            policy_positions=policy_positions,
            logical_tensions=tensions,
            overall_consistency_score=consistency_score
        )

    async def _create_candidate_policy_profiles(self, candidates: List, election_analysis) -> List[PersonPolicyProfile]:
        """Create policy profiles for all candidates"""

        candidate_profiles = []

        for candidate in candidates:
            profile = await self._create_single_candidate_profile(candidate, election_analysis)
            candidate_profiles.append(profile)

        return candidate_profiles

    async def _create_single_candidate_profile(self, candidate, election_analysis) -> PersonPolicyProfile:
        """Create a policy profile for a single candidate"""

        policy_positions = []

        # Create mapping lookup
        question_to_dimension = {}
        for mapping in election_analysis.question_mappings:
            question_to_dimension[mapping.question] = mapping

        self.logger.debug(
            f"Processing candidate {candidate.candidate_id}: "
            f"{len(candidate.responses)} responses, "
            f"{len(question_to_dimension)} mapped questions"
        )

        # Process each candidate response
        responses_processed = 0
        responses_skipped_no_mapping = 0
        responses_skipped_no_dimension = 0
        responses_skipped_none_position = 0
        
        for response in candidate.responses:
            question = response.question
            responses_processed += 1

            if question in question_to_dimension:
                mapping = question_to_dimension[question]

                # Find dimension
                dimension = next(
                    (d for d in election_analysis.discovered_dimensions if d.dimension_id == mapping.primary_dimension_id),
                    None
                )

                if dimension:
                    # Infer position with comment enhancement
                    position = await position_inference_service.infer_policy_position(
                        question=question,
                        answer=response.answer,
                        comment=getattr(response, 'comment', ''),
                        dimension=dimension,
                        person_type="candidate"
                    )
                    if position is not None:
                        policy_positions.append(position)
                    else:
                        responses_skipped_none_position += 1
                        self.logger.debug(
                            f"Candidate {candidate.candidate_id} response #{responses_processed}: "
                            f"Position inference returned None for question '{question[:60]}...' "
                            f"with answer='{str(response.answer)[:50]}...'"
                        )
                else:
                    responses_skipped_no_dimension += 1
                    self.logger.warning(
                        f"Candidate {candidate.candidate_id} response #{responses_processed}: "
                        f"Dimension '{mapping.primary_dimension_id}' not found for question '{question[:60]}...'"
                    )
                    # Handle secondary dimensions
                    for sec_dim_id in mapping.secondary_dimension_ids:
                        sec_dimension = next(
                            (d for d in election_analysis.discovered_dimensions if d.dimension_id == sec_dim_id),
                            None
                        )
                        if sec_dimension:
                            sec_weight = mapping.secondary_weights.get(
                                sec_dim_id, matching_config.position_basic_weight
                            )
                            sec_position = await position_inference_service.infer_policy_position(
                                question=question,
                                answer=response.answer,
                                comment=getattr(response, 'comment', ''),
                                dimension=sec_dimension,
                                person_type="candidate"
                            )
                            if sec_position is not None:
                                sec_position.confidence *= sec_weight
                                sec_position.intensity_multiplier *= sec_weight
                                policy_positions.append(sec_position)

        # Log processing summary for diagnostics
        self.logger.info(
            f"Candidate {candidate.candidate_id} response processing summary: "
            f"Total={responses_processed}, "
            f"Positions created={len(policy_positions)}, "
            f"Skipped (no mapping)={responses_skipped_no_mapping}, "
            f"Skipped (no dimension)={responses_skipped_no_dimension}, "
            f"Skipped (None returned)={responses_skipped_none_position}"
        )
        
        # Analyze consistency
        tensions, consistency_score = consistency_analyzer_service.analyze_position_consistency(policy_positions)

        # Log candidate dimension coverage for debugging zero overlap issues
        candidate_dimension_ids = sorted(set(pos.dimension_id for pos in policy_positions))
        self.logger.info(
            f"Candidate {candidate.candidate_id} profile created with {len(policy_positions)} positions across "
            f"{len(candidate_dimension_ids)} unique dimensions: {candidate_dimension_ids}"
        )
        
        # Log any low-confidence positions that might be filtered out later
        low_confidence_positions = [
            pos for pos in policy_positions 
            if pos.confidence < matching_config.min_confidence_threshold
        ]
        if low_confidence_positions:
            self.logger.warning(
                f"Candidate {candidate.candidate_id} has {len(low_confidence_positions)} positions below confidence threshold "
                f"({matching_config.min_confidence_threshold}): "
                f"{[pos.dimension_id for pos in low_confidence_positions]}"
            )

        return PersonPolicyProfile(
            person_id=candidate.candidate_id,
            person_type="candidate",
            policy_positions=policy_positions,
            logical_tensions=tensions,
            overall_consistency_score=consistency_score
        )

    def _convert_to_output_format(self, enhanced_result: EnhancedMatchResult):
        """Convert an enhanced match result to legacy format with LLM descriptions"""

        from app.schemas.voters_schema import CandidateMatchSchema, IssueMatchDetailSchema

        # Convert dimension matches to issue matches
        issue_matches = []
        for dim_match in enhanced_result.dimension_matches:
            issue_match = IssueMatchDetailSchema(
                issue=dim_match.dimension_name,
                alignment=self._get_alignment_level(dim_match.alignment_score),
                alignment_score=dim_match.alignment_score,
                voter_position=dim_match.voter_position_description or "No description available",
                candidate_position=dim_match.candidate_position_description or "No description available",
                explanation=dim_match.alignment_explanation
            )
            issue_matches.append(issue_match)

        # Limit top-aligned issues to 3
        top_aligned_issues = enhanced_result.top_aligned_dimensions[:3]

        return CandidateMatchSchema(
            candidate_id=enhanced_result.candidate_id,
            match_percentage=enhanced_result.overall_match_percentage,
            match_strength_visual=enhanced_result.overall_match_percentage / 100.0,
            match_category="PENDING",  # Will be assigned later in _assign_match_categories
            top_aligned_issues=top_aligned_issues,
            issue_matches=issue_matches,
            overall_explanation=enhanced_result.match_explanation
        )

    async def _generate_voter_values_profile(
            self,
            voter_profile: PersonPolicyProfile,
            election_analysis
    ) -> List[VoterValueProfileSchema]:
        """Generate voter values profile from policy positions using LLM"""

        values_profile = []

        # Group positions by dimension for analysis
        dimension_positions = {}
        for position in voter_profile.policy_positions:
            if position.dimension_id not in dimension_positions:
                dimension_positions[position.dimension_id] = []
            dimension_positions[position.dimension_id].append(position)

        # Create a value profile for each dimension where a voter has strong positions
        for dimension_id, positions in dimension_positions.items():
            # Find dimension info
            dimension = next(
                (d for d in election_analysis.discovered_dimensions if d.dimension_id == dimension_id),
                None
            )

            if not dimension:
                continue

            # Calculate average position and intensity
            avg_position = sum(p.position_score for p in positions) / len(positions)
            avg_intensity = sum(p.intensity_multiplier for p in positions) / len(positions)

            # Only include if voter has meaningful position (not neutral)
            if abs(avg_position - 50) > matching_config.voter_profile_neutrality_threshold:

                # Determine priority level based on intensity and position strength
                if avg_intensity >= 1.5 and abs(avg_position - 50) > 25:
                    priority = "High"
                elif avg_intensity >= 1.0 or abs(avg_position - 50) > 20:
                    priority = "Medium"
                else:
                    priority = "Low"

                # Generate LLM description
                description = await self._generate_llm_voter_value_description(
                    dimension, avg_position, positions, priority
                )

                value_item = VoterValueProfileSchema(
                    issue=dimension.name,
                    description=description,
                    priority_level=priority
                )
                values_profile.append(value_item)

        # Sort by priority (High -> Medium -> Low)
        priority_order = {"High": 3, "Medium": 2, "Low": 1}
        values_profile.sort(key=lambda x: priority_order[x.priority_level], reverse=True)

        return values_profile[:matching_config.voter_profile_max_items]

    async def _generate_llm_voter_value_description(
            self,
            dimension,
            avg_position: float,
            positions: List[PolicyPosition],
            priority: str
    ) -> str:
        """
        Use LLM to generate natural, personalized voter value descriptions
        """

        # Check cache first — use hashlib.md5 for consistency across restarts
        reasoning_blob = "|".join(p.reasoning for p in positions)
        reasoning_hash = hashlib.md5(reasoning_blob.encode()).hexdigest()
        cache_key = f"voter_value_desc:{dimension.dimension_id}:{avg_position:.1f}:{priority}:{reasoning_hash}"
        cached_desc = await cache_service.get(cache_key)
        if cached_desc:
            return cached_desc

        # Gather context from all positions in this dimension
        position_contexts = []
        for pos in positions:
            if pos.reasoning and len(pos.reasoning.strip()) > 5:
                position_contexts.append({
                    "question": pos.source_question,
                    "answer": pos.source_answer,
                    "reasoning": pos.reasoning,
                    "intensity": pos.intensity_multiplier
                })

        # Create a context summary
        if position_contexts:
            contexts_text = "\n".join([
                f"- Question: {ctx['question'][:80]}...\n  Answer: {ctx['answer']}\n  Reasoning: {ctx['reasoning'][:100]}..."
                for ctx in position_contexts[:3]  # Limit to 3 most relevant
            ])
        else:
            contexts_text = "Direct responses without detailed reasoning provided"

        # Determine stance
        if avg_position >= 75:
            stance = "strongly support"
        elif avg_position >= 60:
            stance = "support"
        elif avg_position >= 40:
            stance = "have mixed feelings about"
        elif avg_position >= 25:
            stance = "have concerns about"
        else:
            stance = "oppose"

        prompt = f"""
        Create a personalized description of this voter's values and priorities regarding {dimension.name}.
        
        Context:
        - Policy Area: {dimension.name}
        - Description: {dimension.description}
        - Voter's overall stance: {stance} ({avg_position}/100)
        - Priority level: {priority}
        - How they responded to questions:
        {contexts_text}
        
        Requirements:
        - Write in second person ("You")
        - 1-2 sentences maximum
        - Explain WHY this matters to them (their motivation/values)
        - Sound personal and conversational
        - Focus on their underlying values, not just policy positions
        - Be specific about what drives their thinking
        
        Examples of good descriptions:
        "You believe every child deserves equal opportunities to succeed, which is why you strongly advocate for expanding access to specialized educational programs."
        "You prioritize fiscal responsibility and want to ensure new programs have sustainable funding before implementation."
        "You value community input and believe local voices should drive decisions that affect neighborhood schools."
        
        Generate a personalized description:
        """

        messages = [
            {"role": "system", "content": "You are an expert at understanding voter motivations and values. Create personal, empathetic descriptions that capture what truly matters to this voter."},
            {"role": "user", "content": prompt}
        ]

        try:
            from app.services.llm_service import llm_service

            response = await llm_service.call_llm(
                messages,
                max_tokens=matching_config.llm_max_tokens_voter_value_description,
                temperature=matching_config.llm_temperature_voter_description,
            )

            # Clean up the response
            description = response.strip().strip('"\'').strip()

            # Ensure it starts with "You"
            if not description.lower().startswith("you"):
                description = "You " + description.lower()

            # Ensure it ends with a period
            if not description.endswith('.'):
                description += '.'

            # Cache the result
            await cache_service.set(cache_key, description, ttl_seconds=matching_config.description_cache_ttl)

            self.logger.debug(f"Generated LLM voter value description: {description[:50]}...")
            return description

        except Exception as e:
            self.logger.error(f"LLM voter value description failed: {str(e)}")
            # Fallback to the basic description
            return self._create_fallback_voter_value_description(dimension, avg_position, positions, priority)

    def _assign_match_categories(self, matches: List):
        """
        Assign match categories to candidates based on their match percentage and ranking.

        Logic:
        - Top 3 candidates with match_percentage > 0 get "TOP"
        - Other candidates with match_percentage > 0 get "OTHER"
        - Candidates with match_percentage = 0 get "UNMATCH" (already assigned)
        """
        # Separate matches with calculated scores from 0% matches
        calculated_matches = [m for m in matches if m.match_percentage > 0]
        zero_matches = [m for m in matches if m.match_percentage == 0]

        # Assign TOP to first N calculated matches (highest percentages)
        for i, match in enumerate(calculated_matches[:matching_config.top_candidate_count]):
            match.match_category = "TOP"
            self.logger.debug(f"Assigned TOP to candidate {match.candidate_id} with {match.match_percentage}% match")

        # Assign OTHER to remaining calculated matches
        for match in calculated_matches[matching_config.top_candidate_count:]:
            match.match_category = "OTHER"
            self.logger.debug(f"Assigned OTHER to candidate {match.candidate_id} with {match.match_percentage}% match")

        # UNMATCH is already assigned to zero matches in _create_zero_match_result
        for match in zero_matches:
            self.logger.debug(f"Candidate {match.candidate_id} already assigned UNMATCH (0% match)")

        # Log the categorization summary
        top_count = len([m for m in matches if m.match_category == "TOP"])
        other_count = len([m for m in matches if m.match_category == "OTHER"])
        unmatch_count = len([m for m in matches if m.match_category == "UNMATCH"])

        self.logger.info(f"Match categorization: {top_count} TOP, {other_count} OTHER, {unmatch_count} UNMATCH")

    @staticmethod
    def _create_fallback_voter_value_description(dimension, avg_position: float, positions: List[PolicyPosition], priority: str) -> str:
        """Fallback description when LLM fails"""

        if avg_position >= 75:
            stance = "strongly support"
        elif avg_position >= 60:
            stance = "support"
        elif avg_position >= 40:
            stance = "have mixed views on"
        elif avg_position >= 25:
            stance = "have concerns about"
        else:
            stance = "oppose"

        # Use reasoning from the strongest position if available
        strongest_pos = max(positions, key=lambda p: p.intensity_multiplier)
        if strongest_pos.reasoning and len(strongest_pos.reasoning) > 10:
            clean_reasoning = strongest_pos.reasoning.strip()
            if len(clean_reasoning) > 60:
                clean_reasoning = clean_reasoning[:57] + "..."
            return f"You {stance} {dimension.name.lower()}, believing that {clean_reasoning.lower()}."
        else:
            return f"You {stance} {dimension.name.lower()} based on your responses to related questions."

    @staticmethod
    def _determine_processing_quality(
        matches: List, 
        election_analysis, 
        eligible_count: int = 0,
        ineligible_count: int = 0
    ) -> tuple[str, float]:
        """Determine processing method and confidence
        
        Args:
            matches: List of candidate matches
            election_analysis: Election analysis data
            eligible_count: Number of candidates that passed eligibility check
            ineligible_count: Number of candidates that failed eligibility check
            
        Returns:
            Tuple of (processing_method, confidence_score)
        """
        
        if not matches:
            return "no_matches", 0.0

        # Separate calculated matches from 0% matches
        calculated_matches = [m for m in matches if m.match_percentage > 0]
        zero_matches = [m for m in matches if m.match_percentage == 0]

        if not calculated_matches:
            # Distinguish between truly ineligible candidates vs eligible with insufficient overlap
            if eligible_count == 0:
                # All candidates failed eligibility check (no profile/questionnaire completion)
                return "no_eligible_candidates", 0.0
            else:
                # Candidates were eligible but all scored 0% due to insufficient dimension overlap
                return "insufficient_dimension_overlap", 0.0

        # Check if we have a good dimension discovery
        avg_dimension_confidence = sum(d.confidence for d in election_analysis.discovered_dimensions) / len(election_analysis.discovered_dimensions)

        # Check match quality (only for calculated matches)
        high_quality_matches = sum(1 for m in calculated_matches if m.match_percentage >= 60)
        has_detailed_explanations = any(m.overall_explanation for m in calculated_matches)

        # Base confidence on dimension discovery quality
        base_confidence = avg_dimension_confidence

        # Adjust confidence based on the proportion of ineligible candidates
        total_candidates = len(matches)
        eligible_ratio = len(calculated_matches) / total_candidates if total_candidates > 0 else 0

        # Reduce confidence if many candidates are ineligible
        confidence_adjustment = eligible_ratio * 0.1  # Up to 10% boost for all eligible

        if has_detailed_explanations and high_quality_matches >= 2:
            return "policy_enhanced", min(0.95, base_confidence + 0.2 + confidence_adjustment)
        elif high_quality_matches >= 1:
            return "policy_based", min(0.85, base_confidence + 0.1 + confidence_adjustment)
        elif calculated_matches:
            return "basic_policy", max(0.6, base_confidence + confidence_adjustment)
        else:
            return "fallback", 0.3

    @staticmethod
    def _create_insufficient_responses_response(
        submission: VoterSubmissionSchema, reason: str
    ) -> MatchResultsResponseSchema:
        """
        Returned when the voter has not provided enough policy-relevant answers.
        The payload shape is identical to a normal response so downstream
        consumers are never broken — matches will simply be empty and the
        processing_method will indicate why.
        """
        return MatchResultsResponseSchema(
            citizen_id=submission.citizen_id,
            election_id=submission.election_id,
            voter_values_profile=[],
            matches=[],
            generated_at=datetime.now(),
            processing_method="insufficient_responses",
            confidence_score=0.0,
        )

    @staticmethod
    def _create_no_candidates_response(submission: VoterSubmissionSchema) -> MatchResultsResponseSchema:
        """Create response when no eligible candidates found"""

        return MatchResultsResponseSchema(
            citizen_id=submission.citizen_id,
            election_id=submission.election_id,
            voter_values_profile=[],
            matches=[],
            generated_at=datetime.now(),
            processing_method="no_candidates",
            confidence_score=0.0
        )

    def _create_error_response(self, submission: VoterSubmissionSchema, error_message: str) -> MatchResultsResponseSchema:
        """Create error response"""

        self.logger.error(f"Creating error response: {error_message}")

        return MatchResultsResponseSchema(
            citizen_id=submission.citizen_id,
            election_id=submission.election_id,
            voter_values_profile=[],
            matches=[],
            generated_at=datetime.now(),
            processing_method="error",
            confidence_score=0.0
        )

    def _create_zero_match_result(self, candidate):
        """Create a 0% match result for candidates with no response data.
        
        Note: With lenient eligibility (responses > 0), this is only called for
        candidates who have truly provided zero responses.
        """
        from app.schemas.voters_schema import CandidateMatchSchema, IssueMatchDetailSchema

        # Determine why this candidate has 0% match
        response_count = len(candidate.responses) if hasattr(candidate, 'responses') else 0
        
        if response_count == 0:
            explanation = (
                "This candidate has not answered any questions yet, "
                "so we cannot compare their policy positions with yours."
            )
        else:
            # Shouldn't happen with new eligibility logic, but handle gracefully
            explanation = (
                "Not enough information available to compare your policy positions "
                "with this candidate's positions."
            )

        return CandidateMatchSchema(
            candidate_id=candidate.candidate_id,
            match_percentage=0,
            match_strength_visual=0.0,
            match_category="UNMATCH",  # Assign UNMATCH category immediately for 0% matches
            top_aligned_issues=[],
            issue_matches=[],
            overall_explanation=explanation
        )

    # Utility methods
    @staticmethod
    def _get_alignment_level(score: float) -> str:
        """Convert score to alignment level"""
        if score >= 0.8:
            return "Strongly Aligned"
        elif score >= 0.6:
            return "Moderately Aligned"
        elif score >= 0.4:
            return "Somewhat Aligned"
        else:
            return "Weakly Aligned"


# Create a service instance
matching_engine = MatchingEngineService()