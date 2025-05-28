# app/services/position_inference_service.py
import re
from typing import Tuple, List, Optional, Any, Dict
from app.services.llm_service import llm_service
from app.services.caching_service import cache_service
from app.schemas.policy_matching_schema import PolicyPosition, PolicyDimension
from app.core.matching_config import matching_config
from app.utils.logging_util import setup_logger

class PositionInferenceService:
    """Service for inferring policy positions from responses"""

    def __init__(self):
        self.logger = setup_logger(__name__)

    async def infer_policy_position(
            self,
            question: str,
            answer: Any,
            comment: str,
            dimension: PolicyDimension,
            person_type: str = "voter"
    ) -> PolicyPosition:
        """
        Main entry point for inferring a person's position on a policy dimension
        """

        # Check cache first
        if matching_config.cache_position_inference:
            cache_key = self._generate_position_cache_key(question, answer, comment, dimension.dimension_id)
            cached_position = await cache_service.get(cache_key)
            if cached_position:
                return PolicyPosition(**cached_position)

        # Step 1: Extract basic position from answer
        basic_position = self._extract_basic_position(answer, dimension)

        # Step 2: Analyze intensity from answer text
        intensity_multiplier = self._calculate_intensity_multiplier(answer)

        # Step 3: Use comment for enhanced inference (mainly for candidates)
        if comment and comment.strip():
            enhanced_position = await self._enhance_position_with_comment(
                basic_position, comment, dimension, question
            )
        else:
            enhanced_position = basic_position

        # Step 4: Calculate confidence based on available information
        confidence = self._calculate_position_confidence(answer, comment, dimension, person_type)

        # Step 5: Extract reasoning
        reasoning = self._extract_reasoning(answer, comment)

        # Create final position
        position = PolicyPosition(
            dimension_id=dimension.dimension_id,
            position_score=enhanced_position["score"],
            confidence=confidence,
            intensity_multiplier=intensity_multiplier,
            reasoning=reasoning,
            source_question=question,
            source_answer=answer
        )

        # Cache the result
        if matching_config.cache_position_inference:
            await cache_service.set(cache_key, position.model_dump(), ttl_seconds=3600)

        return position

    def _extract_basic_position(self, answer: Any, dimension: PolicyDimension) -> Dict[str, float]:
        """
        Extract basic position score from the direct answer
        """

        if isinstance(answer, bool):
            return {"score": 80.0 if answer else 20.0, "method": "boolean"}

        if isinstance(answer, str):
            answer_lower = answer.lower().strip()

            # Strong positive
            if any(phrase in answer_lower for phrase in ["strongly agree", "strongly support", "strongly favor"]):
                return {"score": 90.0, "method": "strong_positive"}

            # Positive
            if any(phrase in answer_lower for phrase in ["agree", "support", "favor", "yes"]):
                return {"score": 75.0, "method": "positive"}

            # Neutral
            if any(phrase in answer_lower for phrase in ["neutral", "unsure", "maybe", "somewhat"]):
                return {"score": 50.0, "method": "neutral"}

            # Negative
            if any(phrase in answer_lower for phrase in ["disagree", "oppose", "against", "no"]):
                return {"score": 25.0, "method": "negative"}

            # Strong negative
            if any(phrase in answer_lower for phrase in ["strongly disagree", "strongly oppose", "strongly against"]):
                return {"score": 10.0, "method": "strong_negative"}

            # Text response - try to infer sentiment
            return self._analyze_text_sentiment(answer_lower)

        # Fallback for other types
        return {"score": 50.0, "method": "fallback"}

    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Basic sentiment analysis for text responses
        """

        positive_indicators = len([w for w in ["good", "great", "excellent", "important", "necessary", "should", "must"] if w in text])
        negative_indicators = len([w for w in ["bad", "poor", "unnecessary", "shouldn't", "cannot", "won't", "refuse"] if w in text])

        if positive_indicators > negative_indicators:
            return {"score": 60.0 + min(positive_indicators * 5, 20), "method": "text_positive"}
        elif negative_indicators > positive_indicators:
            return {"score": 40.0 - min(negative_indicators * 5, 20), "method": "text_negative"}
        else:
            return {"score": 50.0, "method": "text_neutral"}

    def _calculate_intensity_multiplier(self, answer: Any) -> float:
        """
        Calculate intensity multiplier based on answer strength
        """

        if isinstance(answer, str):
            answer_lower = answer.lower().strip()

            # Check for intensity words
            for phrase, multiplier in matching_config.intensity_multipliers.items():
                if phrase.replace("_", " ") in answer_lower:
                    return multiplier

            # Default for text
            return 1.0

        # Boolean answers get standard multiplier
        return 1.0

    async def _enhance_position_with_comment(
            self,
            basic_position: Dict[str, float],
            comment: str,
            dimension: PolicyDimension,
            question: str
    ) -> Dict[str, float]:
        """
        Use LLM to enhance position inference with comment analysis
        """

        prompt = f"""
        Analyze this response to determine the person's position on the policy dimension.
        
        Policy Dimension: {dimension.name}
        Description: {dimension.description}
        Scale: {dimension.policy_spectrum_description}
        
        Question: {question}
        Comment: {comment}
        
        Based on the comment, determine:
        1. Their position on a 0-100 scale where {dimension.policy_spectrum_description}
        2. How confident you are in this assessment (0.0-1.0)
        3. Brief explanation of their reasoning
        
        Consider:
        - Explicit statements of support/opposition
        - Nuanced positions (e.g., "support but with conditions")
        - Intensity of language
        - Specific policy preferences mentioned
        
        Return ONLY a valid JSON object:
        {{
            "position_score": 75.5,
            "confidence": 0.85,
            "reasoning": "Supports the concept but advocates for specific implementation approach"
        }}
        """

        messages = [
            {"role": "system", "content": "You are a policy analyst. Analyze text to infer policy positions accurately. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await llm_service.call_llm(
                messages,
                max_tokens=300,
                temperature=matching_config.position_inference_temperature
            )

            analysis = llm_service._extract_json_from_response(response)

            if analysis and "position_score" in analysis:
                # Blend with basic position (weighted average)
                enhanced_score = (basic_position["score"] * 0.3) + (analysis["position_score"] * 0.7)
                return {
                    "score": enhanced_score,
                    "method": "llm_enhanced",
                    "llm_reasoning": analysis.get("reasoning", "")
                }

        except Exception as e:
            self.logger.error(f"LLM position enhancement failed: {str(e)}")

        # Fallback to basic position
        return basic_position

    def _calculate_position_confidence(
            self,
            answer: Any,
            comment: str,
            dimension: PolicyDimension,
            person_type: str
    ) -> float:
        """
        Calculate confidence in the position inference
        """

        base_confidence = 0.7

        # Adjust based on answer type
        if isinstance(answer, bool):
            base_confidence += 0.1  # Boolean answers are clearer
        elif isinstance(answer, str):
            answer_lower = answer.lower()
            if any(word in answer_lower for word in ["strongly", "definitely", "absolutely"]):
                base_confidence += 0.15
            elif any(word in answer_lower for word in ["maybe", "perhaps", "might"]):
                base_confidence -= 0.2

        # Adjust based on comment availability
        if comment and len(comment.strip()) > 20:
            base_confidence += 0.1  # More context = higher confidence

        # Adjust based on dimension match quality
        if dimension.confidence < 0.7:
            base_confidence -= 0.1  # Less confident dimension mapping

        # Candidate vs voter adjustment
        if person_type == "candidate" and comment:
            base_confidence += 0.05  # Candidates usually give more detailed responses

        return max(0.1, min(1.0, base_confidence))

    def _extract_reasoning(self, answer: Any, comment: str) -> str:
        """
        Extract human-readable reasoning for the position
        """

        if comment and len(comment.strip()) > 10:
            # Use first sentence of comment as reasoning
            sentences = comment.split('.')
            if sentences:
                return sentences[0].strip() + "."

        # Fallback to answer-based reasoning
        if isinstance(answer, bool):
            return "Direct yes/no response"
        elif isinstance(answer, str):
            if len(answer) > 50:
                return answer[:47] + "..."
            return answer
        else:
            return "Selected from available options"

    def _generate_position_cache_key(self, question: str, answer: Any, comment: str, dimension_id: str) -> str:
        """
        Generate cache key for position inference
        """
        import hashlib

        content = f"{question}|{answer}|{comment}|{dimension_id}"
        return f"position_inference:{hashlib.md5(content.encode()).hexdigest()}"

# Create service instance
position_inference_service = PositionInferenceService()
