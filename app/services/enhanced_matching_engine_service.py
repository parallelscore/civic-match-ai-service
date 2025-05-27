# app/services/enhanced_matching_engine_service.py (Updated)
from datetime import datetime
from typing import List

from app.utils.logging_util import setup_logger
from app.schemas.voters_schema import VoterSubmissionSchema, MatchResultsResponseSchema, VoterValueProfileSchema
from app.schemas.policy_matching_schema import PersonPolicyProfile, PolicyPosition, EnhancedMatchResult
from app.services.candidate_service import candidate_service
from app.services.policy_dimension_discovery_service import policy_dimension_discovery_service
from app.services.position_inference_service import position_inference_service
from app.services.consistency_analyzer_service import consistency_analyzer_service
from app.services.enhanced_matching_calculator_service import enhanced_matching_calculator_service
from app.services.caching_service import cache_service

class EnhancedMatchingEngine:
    """
    Main enhanced matching engine that orchestrates the multi-layer matching process
    """

    def __init__(self):
        self.logger = setup_logger(__name__)

    async def process_voter_submission(self, submission: VoterSubmissionSchema) -> MatchResultsResponseSchema:
        """
        Process voter submission using the enhanced policy-based matching system
        """

        self.logger.info(f"Processing enhanced submission for voter {submission.citizen_id} in election {submission.election_id}")

        try:
            # Step 1: Get candidates
            candidates = await candidate_service.get_candidates_for_election(submission.election_id)
            if not candidates:
                return self._create_no_candidates_response(submission)

            # Step 2: Discover policy dimensions for this election
            all_questions = self._extract_all_questions(submission, candidates)
            election_analysis = await policy_dimension_discovery_service.discover_election_policy_dimensions(
                submission.election_id, all_questions
            )

            self.logger.info(f"Discovered {len(election_analysis.discovered_dimensions)} policy dimensions")

            # Step 3: Create voter policy profile
            voter_profile = await self._create_voter_policy_profile(submission, election_analysis)

            # Step 4: Create candidate policy profiles
            candidate_profiles = await self._create_candidate_policy_profiles(candidates, election_analysis)

            # Step 5: Calculate matches
            enhanced_matches = []
            for candidate_profile in candidate_profiles:
                match_result = await enhanced_matching_calculator_service.calculate_enhanced_match(
                    voter_profile, candidate_profile
                )
                enhanced_matches.append(self._convert_to_legacy_format(match_result))

            # Step 6: Sort by match percentage
            enhanced_matches.sort(key=lambda x: x.match_percentage, reverse=True)

            # Step 7: Generate voter values profile
            voter_values_profile = await self._generate_voter_values_profile(voter_profile, election_analysis)

            # Step 8: Determine processing method and confidence
            processing_method, confidence = self._determine_processing_quality(enhanced_matches, election_analysis)

            self.logger.info(f"Processing method: {processing_method}, Confidence: {confidence:.2f}")

            return MatchResultsResponseSchema(
                citizen_id=submission.citizen_id,
                election_id=submission.election_id,
                voter_values_profile=voter_values_profile,
                matches=enhanced_matches,
                generated_at=datetime.now(),
                processing_method=processing_method,
                confidence_score=confidence
            )

        except Exception as e:
            self.logger.error(f"Enhanced matching failed: {str(e)}")
            # Fallback to basic matching or error response
            return self._create_error_response(submission, str(e))

    def _extract_all_questions(self, submission: VoterSubmissionSchema, candidates: List) -> List[str]:
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

    async def _create_voter_policy_profile(
            self,
            submission: VoterSubmissionSchema,
            election_analysis
    ) -> PersonPolicyProfile:
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
                    position = await position_inference_service.infer_policy_position(
                        question=question,
                        answer=response.answer,
                        comment="",  # Voters typically don't have comments
                        dimension=dimension,
                        person_type="voter"
                    )
                    policy_positions.append(position)

                    # Handle secondary dimensions if they exist
                    for sec_dim_id in mapping.secondary_dimension_ids:
                        sec_dimension = next(
                            (d for d in election_analysis.discovered_dimensions if d.dimension_id == sec_dim_id),
                            None
                        )
                        if sec_dimension:
                            # Create weighted position for secondary dimension
                            sec_weight = mapping.secondary_weights.get(sec_dim_id, 0.3)
                            sec_position = await position_inference_service.infer_policy_position(
                                question=question,
                                answer=response.answer,
                                comment="",
                                dimension=sec_dimension,
                                person_type="voter"
                            )
                            # Adjust confidence and intensity for secondary mapping
                            sec_position.confidence *= sec_weight
                            sec_position.intensity_multiplier *= sec_weight
                            policy_positions.append(sec_position)

        # Analyze consistency
        tensions, consistency_score = consistency_analyzer_service.analyze_position_consistency(policy_positions)

        return PersonPolicyProfile(
            person_id=submission.citizen_id,
            person_type="voter",
            policy_positions=policy_positions,
            logical_tensions=tensions,
            overall_consistency_score=consistency_score
        )

    async def _create_candidate_policy_profiles(
            self,
            candidates: List,
            election_analysis
    ) -> List[PersonPolicyProfile]:
        """Create policy profiles for all candidates"""

        candidate_profiles = []

        for candidate in candidates:
            profile = await self._create_single_candidate_profile(candidate, election_analysis)
            candidate_profiles.append(profile)

        return candidate_profiles

    async def _create_single_candidate_profile(
            self,
            candidate,
            election_analysis
    ) -> PersonPolicyProfile:
        """Create policy profile for a single candidate"""

        policy_positions = []

        # Create mapping lookup
        question_to_dimension = {}
        for mapping in election_analysis.question_mappings:
            question_to_dimension[mapping.question] = mapping

        # Process each candidate response
        for response in candidate.responses:
            question = response.question

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
                    policy_positions.append(position)

                    # Handle secondary dimensions
                    for sec_dim_id in mapping.secondary_dimension_ids:
                        sec_dimension = next(
                            (d for d in election_analysis.discovered_dimensions if d.dimension_id == sec_dim_id),
                            None
                        )
                        if sec_dimension:
                            sec_weight = mapping.secondary_weights.get(sec_dim_id, 0.3)
                            sec_position = await position_inference_service.infer_policy_position(
                                question=question,
                                answer=response.answer,
                                comment=getattr(response, 'comment', ''),
                                dimension=sec_dimension,
                                person_type="candidate"
                            )
                            sec_position.confidence *= sec_weight
                            sec_position.intensity_multiplier *= sec_weight
                            policy_positions.append(sec_position)

        # Analyze consistency
        tensions, consistency_score = consistency_analyzer_service.analyze_position_consistency(policy_positions)

        return PersonPolicyProfile(
            person_id=candidate.candidate_id,
            person_type="candidate",
            policy_positions=policy_positions,
            logical_tensions=tensions,
            overall_consistency_score=consistency_score
        )

    def _convert_to_legacy_format(self, enhanced_result: EnhancedMatchResult):
        """Convert enhanced match result to legacy format with LLM descriptions"""

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

        # Limit top aligned issues to 3
        top_aligned_issues = enhanced_result.top_aligned_dimensions[:3]

        return CandidateMatchSchema(
            candidate_id=enhanced_result.candidate_id,
            match_percentage=enhanced_result.overall_match_percentage,
            match_strength_visual=enhanced_result.overall_match_percentage / 100.0,
            top_aligned_issues=top_aligned_issues,
            issue_matches=issue_matches,
            overall_explanation=enhanced_result.match_explanation
        )

    def _create_detailed_position_description(self, position: PolicyPosition) -> str:
        """
        Create detailed position description (same as in calculator service)
        """
        score = position.position_score
        reasoning = position.reasoning if position.reasoning and len(position.reasoning.strip()) > 5 else ""

        # Base position strength
        if score >= 85:
            base_desc = "Very strong support"
        elif score >= 70:
            base_desc = "Strong support"
        elif score >= 60:
            base_desc = "Moderate support"
        elif score >= 40:
            base_desc = "Mixed views"
        elif score >= 25:
            base_desc = "Moderate opposition"
        else:
            base_desc = "Strong opposition"

        # Add reasoning context if available
        if reasoning:
            reasoning_lower = reasoning.lower()

            if any(word in reasoning_lower for word in ["access", "opportunity", "program"]):
                context = "focusing on expanding access and opportunities"
            elif any(word in reasoning_lower for word in ["funding", "resource", "budget"]):
                context = "emphasizing adequate funding and resources"
            elif any(word in reasoning_lower for word in ["safety", "security", "protection"]):
                context = "prioritizing safety and security measures"
            elif any(word in reasoning_lower for word in ["mental health", "counseling", "support"]):
                context = "emphasizing comprehensive mental health support"
            elif any(word in reasoning_lower for word in ["community", "local", "resident"]):
                context = "focusing on community engagement and local input"
            elif any(word in reasoning_lower for word in ["quality", "improvement", "standard"]):
                context = "emphasizing quality improvements and higher standards"
            elif any(word in reasoning_lower for word in ["necessary", "need", "important"]):
                context = "viewing this as essential for student success"
            else:
                # Truncate reasoning if too long
                context = f"based on their view that {reasoning[:50]}..."

            return f"{base_desc}, {context}."

        return f"{base_desc} for this policy area."


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

        # Create value profile for each dimension where voter has strong positions
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
            if abs(avg_position - 50) > 15:  # More than 15 points from neutral

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

        return values_profile[:6]

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

        # Check cache first
        cache_key = f"voter_value_desc:{dimension.dimension_id}:{avg_position}:{priority}:{hash(str([p.reasoning for p in positions]))}"
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

        # Create context summary
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

            response = await llm_service._call_llm(
                messages,
                max_tokens=100,
                temperature=0.4
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
            await cache_service.set(cache_key, description, ttl_seconds=3600)

            self.logger.debug(f"Generated LLM voter value description: {description[:50]}...")
            return description

        except Exception as e:
            self.logger.error(f"LLM voter value description failed: {str(e)}")
            # Fallback to basic description
            return self._create_fallback_voter_value_description(dimension, avg_position, positions, priority)

    def _create_fallback_voter_value_description(
            self,
            dimension,
            avg_position: float,
            positions: List[PolicyPosition],
            priority: str
    ) -> str:
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

        # Use reasoning from strongest position if available
        strongest_pos = max(positions, key=lambda p: p.intensity_multiplier)
        if strongest_pos.reasoning and len(strongest_pos.reasoning) > 10:
            clean_reasoning = strongest_pos.reasoning.strip()
            if len(clean_reasoning) > 60:
                clean_reasoning = clean_reasoning[:57] + "..."
            return f"You {stance} {dimension.name.lower()}, believing that {clean_reasoning.lower()}."
        else:
            return f"You {stance} {dimension.name.lower()} based on your responses to related questions."



    def _generate_value_description(
            self,
            avg_position: float,
            dimension,
            positions: List[PolicyPosition]
    ) -> str:
        """Generate description for a voter value"""

        # Determine position strength
        if avg_position >= 75:
            strength = "strongly support"
        elif avg_position >= 60:
            strength = "support"
        elif avg_position >= 40:
            strength = "have mixed views on"
        elif avg_position >= 25:
            strength = "have concerns about"
        else:
            strength = "oppose"

        # Use reasoning from strongest position if available
        strongest_pos = max(positions, key=lambda p: p.intensity_multiplier)
        if strongest_pos.reasoning and len(strongest_pos.reasoning) > 10:
            return f"You {strength} {dimension.name.lower()}. {strongest_pos.reasoning}"
        else:
            return f"You {strength} {dimension.name.lower()} based on your responses to related questions."

    def _determine_processing_quality(
            self,
            matches: List,
            election_analysis
    ) -> tuple[str, float]:
        """Determine processing method and confidence"""

        if not matches:
            return "no_matches", 0.0

        # Check if we have good dimension discovery
        avg_dimension_confidence = sum(d.confidence for d in election_analysis.discovered_dimensions) / len(election_analysis.discovered_dimensions)

        # Check match quality
        high_quality_matches = sum(1 for m in matches if m.match_percentage >= 60)
        has_detailed_explanations = any(m.overall_explanation for m in matches)

        # Base confidence on dimension discovery quality
        base_confidence = avg_dimension_confidence

        if has_detailed_explanations and high_quality_matches >= 2:
            return "policy_enhanced", min(0.95, base_confidence + 0.2)
        elif high_quality_matches >= 1:
            return "policy_based", min(0.85, base_confidence + 0.1)
        elif matches:
            return "basic_policy", max(0.6, base_confidence)
        else:
            return "fallback", 0.3

    def _create_no_candidates_response(self, submission: VoterSubmissionSchema) -> MatchResultsResponseSchema:
        """Create response when no candidates found"""

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

    @staticmethod
    def _format_position(position: PolicyPosition) -> str:
        """Format position for display"""
        if position.position_score >= 75:
            return "Strong support"
        elif position.position_score >= 60:
            return "Support"
        elif position.position_score >= 40:
            return "Mixed/Neutral"
        elif position.position_score >= 25:
            return "Opposition"
        else:
            return "Strong opposition"

# Create service instance
enhanced_matching_engine = EnhancedMatchingEngine()