# app/services/enhanced_matching_calculator_service.py
import hashlib
from typing import List
from app.schemas.policy_matching_schema import (
    PolicyPosition, DimensionMatch, EnhancedMatchResult,
    PersonPolicyProfile
)
from app.services.consistency_analyzer_service import consistency_analyzer_service
from app.core.matching_config import matching_config
from app.utils.logging_util import setup_logger
from app.services.caching_service import cache_service

class EnhancedMatchingCalculatorService:
    """Service for calculating enhanced policy matches between voters and candidates"""

    def __init__(self):
        self.logger = setup_logger(__name__)

    async def calculate_enhanced_match(
            self,
            voter_profile: PersonPolicyProfile,
            candidate_profile: PersonPolicyProfile
    ) -> EnhancedMatchResult:
        """
        Calculate comprehensive match between voter and candidate
        """

        self.logger.info(f"Calculating enhanced match between voter {voter_profile.person_id} and candidate {candidate_profile.person_id}")

        # Step 1: Calculate matches for each policy dimension
        dimension_matches = await self._calculate_dimension_matches(
            voter_profile.policy_positions,
            candidate_profile.policy_positions
        )

        # Step 1a: Minimum overlap guard — if voter and candidate share too few
        # comparable dimensions, a score would be misleading. Return 0% instead.
        if len(dimension_matches) < matching_config.min_dimension_overlap:
            self.logger.info(
                f"Candidate {candidate_profile.person_id} has only "
                f"{len(dimension_matches)} overlapping dimension(s) "
                f"(minimum {matching_config.min_dimension_overlap}) — returning 0%"
            )
            return EnhancedMatchResult(
                voter_id=voter_profile.person_id,
                candidate_id=candidate_profile.person_id,
                overall_match_percentage=0,
                confidence_weighted_percentage=0,
                dimension_matches=dimension_matches,
                consistency_penalty_applied=0.0,
                match_explanation=(
                    "Not enough comparable policy positions found between your "
                    "responses and this candidate to calculate a meaningful match."
                ),
                top_aligned_dimensions=[],
            )

        # Step 2: Calculate base match percentage
        base_match_percentage = self._calculate_base_match_percentage(dimension_matches)

        # Step 3: Apply consistency penalties
        consistency_penalty = self._calculate_consistency_penalty(
            voter_profile.overall_consistency_score,
            candidate_profile.overall_consistency_score
        )

        adjusted_match_percentage = consistency_analyzer_service.apply_consistency_penalty(
            base_match_percentage,
            min(voter_profile.overall_consistency_score, candidate_profile.overall_consistency_score)
        )

        # Step 4: Calculate confidence-weighted percentage
        confidence_weighted_percentage = self._calculate_confidence_weighted_percentage(dimension_matches)

        # Step 5: Generate explanations
        match_explanation = self._generate_match_explanation(
            dimension_matches,
            base_match_percentage,
            adjusted_match_percentage,
            consistency_penalty
        )

        # Step 6: Identify top aligned dimensions
        top_aligned_dimensions = self._identify_top_aligned_dimensions(dimension_matches)

        # Create final result
        result = EnhancedMatchResult(
            voter_id=voter_profile.person_id,
            candidate_id=candidate_profile.person_id,
            overall_match_percentage=int(adjusted_match_percentage),
            confidence_weighted_percentage=int(confidence_weighted_percentage),
            dimension_matches=dimension_matches,
            consistency_penalty_applied=consistency_penalty,
            match_explanation=match_explanation,
            top_aligned_dimensions=top_aligned_dimensions
        )

        self.logger.info(f"Match result: {result.overall_match_percentage}% (confidence-weighted: {result.confidence_weighted_percentage}%)")

        return result

    async def _calculate_dimension_matches(
            self,
            voter_positions: List[PolicyPosition],
            candidate_positions: List[PolicyPosition]
    ) -> List[DimensionMatch]:
        """
        Calculate match for each policy dimension with LLM-generated descriptions
        """

        # Create lookup dictionaries — filter out positions below the minimum
        # confidence threshold so that very uncertain inferences never feed
        # into the final match score.
        voter_lookup = {
            pos.dimension_id: pos
            for pos in voter_positions
            if pos.confidence >= matching_config.min_confidence_threshold
        }
        candidate_lookup = {
            pos.dimension_id: pos
            for pos in candidate_positions
            if pos.confidence >= matching_config.min_confidence_threshold
        }

        if len(voter_positions) != len(voter_lookup):
            self.logger.debug(
                f"Filtered out {len(voter_positions) - len(voter_lookup)} low-confidence "
                f"voter positions (threshold={matching_config.min_confidence_threshold})"
            )
        if len(candidate_positions) != len(candidate_lookup):
            self.logger.debug(
                f"Filtered out {len(candidate_positions) - len(candidate_lookup)} low-confidence "
                f"candidate positions (threshold={matching_config.min_confidence_threshold})"
            )

        dimension_matches = []

        # Find common dimensions
        common_dimensions = set(voter_lookup.keys()) & set(candidate_lookup.keys())

        for dimension_id in common_dimensions:
            voter_pos = voter_lookup[dimension_id]
            candidate_pos = candidate_lookup[dimension_id]

            # Calculate raw alignment score
            alignment_score = self._calculate_position_alignment(voter_pos, candidate_pos)

            # Apply confidence weighting
            confidence_weighted_score = self._apply_confidence_weighting(
                alignment_score,
                voter_pos.confidence,
                candidate_pos.confidence
            )

            # Generate LLM-based alignment explanation and position descriptions
            dimension_name = dimension_id.replace('_', ' ').title()
            alignment_explanation, voter_desc, candidate_desc = await self._generate_alignment_explanation(
                voter_pos,
                candidate_pos,
                alignment_score,
                dimension_name
            )

            dimension_match = DimensionMatch(
                dimension_id=dimension_id,
                dimension_name=dimension_name,
                alignment_score=alignment_score,
                confidence_weighted_score=confidence_weighted_score,
                voter_position=voter_pos,
                candidate_position=candidate_pos,
                alignment_explanation=alignment_explanation,
                voter_position_description=voter_desc,      # Add this field
                candidate_position_description=candidate_desc  # Add this field
            )

            dimension_matches.append(dimension_match)

        self.logger.debug(f"Calculated matches for {len(dimension_matches)} common dimensions")

        return dimension_matches

    def _calculate_position_alignment(self, voter_pos: PolicyPosition, candidate_pos: PolicyPosition) -> float:
        """
        Calculate alignment score between two policy positions
        """

        # Calculate raw distance on 0-100 scale
        position_distance = abs(voter_pos.position_score - candidate_pos.position_score)

        # Convert distance to similarity (0-1 scale)
        base_similarity = max(0.0, (100 - position_distance) / 100)

        # Apply intensity weighting - stronger voter positions matter more
        intensity_weight = (voter_pos.intensity_multiplier + 1.0) / 3.0  # Normalize to 0.33-1.0
        weighted_similarity = base_similarity * intensity_weight

        # Boost for high agreement areas (configurable threshold and factor)
        distance = abs(voter_pos.position_score - candidate_pos.position_score)
        if distance < matching_config.high_agreement_distance_threshold:
            weighted_similarity = min(1.0, weighted_similarity * matching_config.high_agreement_boost)

        self.logger.debug(f"Position alignment: distance={position_distance:.1f}, base_sim={base_similarity:.3f}, weighted_sim={weighted_similarity:.3f}")

        return weighted_similarity

    def _apply_confidence_weighting(self, alignment_score: float, voter_confidence: float, candidate_confidence: float) -> float:
        """
        Apply confidence weighting to alignment score
        """

        if not matching_config.confidence_weighting_enabled:
            return alignment_score

        # Use minimum confidence as the limiting factor
        combined_confidence = min(voter_confidence, candidate_confidence)

        # Apply confidence weighting
        confidence_weighted_score = alignment_score * combined_confidence

        return confidence_weighted_score

    def _calculate_base_match_percentage(self, dimension_matches: List[DimensionMatch]) -> float:
        """
        Calculate base match percentage from dimension matches
        """

        if not dimension_matches:
            return 0.0

        # Weight by voter intensity - dimensions where voter feels strongly matter more
        total_weighted_score = 0.0
        total_weight = 0.0

        for match in dimension_matches:
            weight = match.voter_position.intensity_multiplier
            weighted_score = match.alignment_score * weight

            total_weighted_score += weighted_score
            total_weight += weight

        # Calculate weighted average
        if total_weight > 0:
            base_percentage = (total_weighted_score / total_weight) * 100
        else:
            base_percentage = sum(match.alignment_score for match in dimension_matches) / len(dimension_matches) * 100

        return max(0.0, min(100.0, base_percentage))

    def _calculate_confidence_weighted_percentage(self, dimension_matches: List[DimensionMatch]) -> float:
        """
        Calculate confidence-weighted match percentage
        """

        if not dimension_matches:
            return 0.0

        if not matching_config.confidence_weighting_enabled:
            return self._calculate_base_match_percentage(dimension_matches)

        # Weight by both voter intensity and confidence
        total_weighted_score = 0.0
        total_weight = 0.0

        for match in dimension_matches:
            intensity_weight = match.voter_position.intensity_multiplier
            confidence_score = match.confidence_weighted_score

            weighted_score = confidence_score * intensity_weight

            total_weighted_score += weighted_score
            total_weight += intensity_weight

        # Calculate weighted average
        if total_weight > 0:
            confidence_percentage = (total_weighted_score / total_weight) * 100
        else:
            confidence_percentage = sum(match.confidence_weighted_score for match in dimension_matches) / len(dimension_matches) * 100

        return max(0.0, min(100.0, confidence_percentage))

    def _calculate_consistency_penalty(self, voter_consistency: float, candidate_consistency: float) -> float:
        """
        Calculate consistency penalty factor
        """

        if not matching_config.enable_consistency_analysis:
            return 0.0

        # Use the lower consistency score as the limiting factor
        min_consistency = min(voter_consistency, candidate_consistency)

        # Calculate penalty (higher penalty for lower consistency)
        penalty = matching_config.consistency_penalty_rate * (1.0 - min_consistency)

        return penalty

    async def _generate_alignment_explanation(
            self,
            voter_pos: PolicyPosition,
            candidate_pos: PolicyPosition,
            alignment_score: float,
            dimension_name: str
    ) -> str:
        """
        Generate human-readable explanation for dimension alignment using LLM-generated descriptions
        """

        position_diff = abs(voter_pos.position_score - candidate_pos.position_score)

        if alignment_score >= 0.8:
            strength = "Strong"
        elif alignment_score >= 0.6:
            strength = "Good"
        elif alignment_score >= 0.4:
            strength = "Moderate"
        else:
            strength = "Weak"

        # Generate LLM-based position descriptions
        try:
            voter_desc = await self._generate_llm_position_description(voter_pos, dimension_name, "voter")
            candidate_desc = await self._generate_llm_position_description(candidate_pos, dimension_name, "candidate")

            # Create alignment explanation
            if position_diff <= 15:
                explanation = f"{strength} alignment: Both of you share similar views on this issue"
            elif position_diff <= 30:
                explanation = f"{strength} alignment: You have somewhat different approaches but generally align"
            else:
                explanation = f"{strength} alignment: You have different perspectives on this issue"

        except Exception as e:
            self.logger.error(f"Failed to generate LLM descriptions: {str(e)}")
            # Fallback to basic explanation
            explanation = f"{strength} alignment based on policy position comparison"
            voter_desc = self._create_fallback_position_description(voter_pos, "voter")
            candidate_desc = self._create_fallback_position_description(candidate_pos, "candidate")

        # Add a confidence note if low
        avg_confidence = (voter_pos.confidence + candidate_pos.confidence) / 2
        if avg_confidence < 0.6:
            explanation += " (assessment has limited confidence due to different question types)"

        return explanation, voter_desc, candidate_desc

    def _generate_match_explanation(
            self,
            dimension_matches: List[DimensionMatch],
            base_percentage: float,
            adjusted_percentage: float,
            consistency_penalty: float
    ) -> str:
        """
        Generate overall match explanation
        """

        if not dimension_matches:
            return "No comparable policy positions found between your responses and this candidate."

        # Categorize the match
        if adjusted_percentage >= 80:
            overall_assessment = "Excellent alignment"
        elif adjusted_percentage >= 70:
            overall_assessment = "Strong alignment"
        elif adjusted_percentage >= 60:
            overall_assessment = "Good alignment"
        elif adjusted_percentage >= 45:
            overall_assessment = "Moderate alignment"
        else:
            overall_assessment = "Limited alignment"

        explanation_parts = [
            f"{overall_assessment} across {len(dimension_matches)} policy areas."
        ]

        # Add top alignment areas
        strong_matches = [m for m in dimension_matches if m.alignment_score >= 0.7]
        if strong_matches:
            strong_areas = [m.dimension_name for m in strong_matches[:3]]
            explanation_parts.append(f"Particularly strong agreement on {', '.join(strong_areas)}.")

        # Add a consistency penalty note
        if consistency_penalty > 0.1:
            explanation_parts.append("Some internal tensions in policy positions detected.")

        # Add confidence note
        low_confidence_matches = [m for m in dimension_matches if m.confidence_weighted_score < m.alignment_score * 0.8]
        if len(low_confidence_matches) > len(dimension_matches) * 0.5:
            explanation_parts.append("Assessment confidence is limited due to different question types.")

        return " ".join(explanation_parts)

    def _identify_top_aligned_dimensions(self, dimension_matches: List[DimensionMatch]) -> List[str]:
        """
        Identify top aligned dimensions for highlighting
        """

        # Sort by alignment score, take top 5
        sorted_matches = sorted(dimension_matches, key=lambda x: x.alignment_score, reverse=True)

        top_dimensions = []
        for match in sorted_matches[:3]:
            if match.alignment_score >= 0.5:  # Only include reasonably aligned dimensions
                top_dimensions.append(match.dimension_name)

        return top_dimensions

    async def _generate_llm_position_description(
            self,
            position: PolicyPosition,
            dimension_name: str,
            person_type: str = "voter"
    ) -> str:
        """
        Use LLM to generate dynamic, human-like position descriptions
        """

        # Check cache first — use hashlib.md5 for cross-restart consistency
        reasoning_hash = hashlib.md5((position.reasoning or "").encode()).hexdigest()
        cache_key = f"position_desc:{position.dimension_id}:{position.position_score:.1f}:{reasoning_hash}"
        cached_desc = await cache_service.get(cache_key)
        if cached_desc:
            return cached_desc

        # Create context for the LLM
        score_context = self._get_score_context(position.position_score)
        reasoning_context = position.reasoning if position.reasoning and len(position.reasoning.strip()) > 5 else "No additional reasoning provided"

        prompt = f"""
        Create a natural, human-like description of this person's position on {dimension_name}.
        
        Context:
        - Position score: {position.position_score}/100 ({score_context})
        - Person type: {person_type}
        - Their reasoning: "{reasoning_context}"
        - Intensity level: {position.intensity_multiplier}x (higher = more passionate)
        
        Requirements:
        - Maximum 2 short sentences
        - Sound natural and conversational
        - Capture both the stance AND the reasoning/motivation
        - Use "you" for voters, "they" for candidates
        - Be specific about their approach or priorities
        
        Examples of good descriptions:
        - "You strongly believe schools need more mental health counselors because students are struggling with unprecedented stress levels."
        - "They support language immersion programs but think the focus should be on improving existing schools first."
        - "You're concerned that adding more programs without proper funding will stretch resources too thin."
        
        Generate a description for this {person_type}'s position:
        """

        messages = [
            {"role": "system", "content": "You are an expert at translating policy positions into natural, human language. Be conversational and specific."},
            {"role": "user", "content": prompt}
        ]

        try:
            from app.services.llm_service import llm_service

            response = await llm_service.call_llm(
                messages,
                max_tokens=matching_config.llm_max_tokens_position_description,
                temperature=matching_config.llm_temperature_position_description,
            )

            # Clean up the response
            description = response.strip().strip('"\'')

            # Ensure it's not too long
            if len(description) > 200:
                sentences = description.split('. ')
                description = '. '.join(sentences[:2])
                if not description.endswith('.'):
                    description += '.'

            # Cache the result
            await cache_service.set(cache_key, description, ttl_seconds=matching_config.description_cache_ttl)

            return description

        except Exception as e:
            self.logger.error(f"LLM position description failed: {str(e)}")
            # Fallback to basic description
            return self._create_fallback_position_description(position, person_type)

    def _get_score_context(self, score: float) -> str:
        """Get contextual description of the score"""
        if score >= 85:
            return "very strong support"
        elif score >= 70:
            return "strong support"
        elif score >= 60:
            return "moderate support"
        elif score >= 40:
            return "mixed/neutral"
        elif score >= 25:
            return "moderate opposition"
        else:
            return "strong opposition"

    def _create_fallback_position_description(self, position: PolicyPosition, person_type: str) -> str:
        """Fallback description when LLM fails"""
        pronoun = "You" if person_type == "voter" else "They"
        score_desc = self._get_score_context(position.position_score)

        if position.reasoning and len(position.reasoning.strip()) > 5:
            return f"{pronoun} show {score_desc} based on your belief that {position.reasoning[:50]}..."
        else:
            return f"{pronoun} show {score_desc} for this policy area."

# Create service instance
enhanced_matching_calculator_service = EnhancedMatchingCalculatorService()