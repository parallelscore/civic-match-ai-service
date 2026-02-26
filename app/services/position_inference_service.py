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
    ) -> Optional[PolicyPosition]:
        """
        Main entry point for inferring a person's position on a policy dimension.

        Pipeline:
          1. Cache check
          2. Basic position extraction (phrase matching / boolean)
          3. Context-aware LLM interpretation for structured short answers
             (yes/no/true/false) — the question is used so the AI knows WHAT
             the yes/no is answering, not just that they said yes/no
          4. Comment enhancement (when a written explanation is provided)
          5. Confidence calculation
          6. Reasoning extraction
          7. Cache store

        Returns None if the answer carries no policy-relevant content.
        """

        # ── Step 1: Cache check ─────────────────────────────────────────
        cache_key = None
        if matching_config.cache_position_inference:
            cache_key = self._generate_position_cache_key(
                question, answer, comment, dimension.dimension_id
            )
            cached_position = await cache_service.get(cache_key)
            if cached_position:
                return PolicyPosition(**cached_position)

        # ── Step 2: Basic position extraction ──────────────────────────
        basic_position = self._extract_basic_position(answer, dimension)

        # Skip answers that carry no policy meaning (name, age, pure numbers…)
        if basic_position["method"] == "irrelevant":
            return None

        # ── Step 3: Context-aware interpretation for structured answers ─
        # For yes/no/true/false answers, the raw score (75 or 25) is just a
        # guess without knowing what question was asked.  We pass the question
        # to the LLM so it can derive the *actual* directional meaning.
        # e.g. "yes" to "Do you support cutting the police budget?" means
        # something very different to "yes" to "Do you support more funding
        # for mental health services?"
        if self._is_structured_answer(answer):
            context_position = await self._interpret_structured_answer_with_context(
                question=question,
                answer=answer,
                basic_position=basic_position,
                dimension=dimension,
            )
            enhanced_position = context_position
        else:
            enhanced_position = basic_position

        # ── Step 4: Comment enhancement ────────────────────────────────
        # If a written comment/explanation exists, blend it in (mainly for candidates)
        if comment and comment.strip():
            enhanced_position = await self._enhance_position_with_comment(
                enhanced_position, comment, dimension, question
            )

        # ── Step 5: Intensity multiplier ───────────────────────────────
        intensity_multiplier = self._calculate_intensity_multiplier(answer)

        # ── Step 6: Confidence ─────────────────────────────────────────
        confidence = self._calculate_position_confidence(answer, comment, dimension, person_type)

        # ── Step 7: Reasoning ──────────────────────────────────────────
        reasoning = self._extract_reasoning(answer, comment, enhanced_position)

        # ── Build result ───────────────────────────────────────────────
        position = PolicyPosition(
            dimension_id=dimension.dimension_id,
            position_score=enhanced_position["score"],
            confidence=confidence,
            intensity_multiplier=intensity_multiplier,
            reasoning=reasoning,
            source_question=question,
            source_answer=answer,
        )

        # ── Cache store ────────────────────────────────────────────────
        if matching_config.cache_position_inference and cache_key:
            await cache_service.set(
                cache_key, position.model_dump(),
                ttl_seconds=matching_config.position_cache_ttl,
            )

        return position

    @staticmethod
    def _is_structured_answer(answer: Any) -> bool:
        """
        Returns True if the answer needs question context to be interpreted
        meaningfully. This covers:

          • Booleans (True/False)
          • Short text tokens (yes/no/agree/disagree etc.)
          • Lists and dicts — multi-select / ranked-choice answers where the
            selected options only make sense when read alongside the question.

        All of these are sent through context-aware LLM interpretation so the
        AI understands WHAT is being agreed/selected, not just the raw value.
        """
        if isinstance(answer, bool):
            return True
        if isinstance(answer, (list, dict)):
            return True  # Multi-select answers always need question context
        if isinstance(answer, str):
            normalized = answer.lower().strip()
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
            return normalized in structured_tokens
        return False

    async def _interpret_structured_answer_with_context(
            self,
            question: str,
            answer: Any,
            basic_position: Dict[str, float],
            dimension: PolicyDimension,
    ) -> Dict[str, float]:
        """
        Use the LLM to interpret a short structured answer (yes/no/true/false/
        agree/disagree) in the context of the specific question being asked.

        This is the core fix for the "yes means nothing without context" problem.

        Example:
          Question: "Do you support reducing the police budget?"
          Answer:   "yes"
          → Without context: score = 75 (positive)
          → With context:    score = 15 (supports reducing police budget →
                             low on safety/security dimension)

        The basic_position is used as a fallback if the LLM call fails.
        """
        # Serialise the answer into a readable string for the LLM prompt.
        # Lists and dicts are formatted clearly so the AI can read the selections.
        if isinstance(answer, bool):
            answer_text = "yes" if answer else "no"
        elif isinstance(answer, list):
            answer_text = ", ".join(str(item) for item in answer)
        elif isinstance(answer, dict):
            answer_text = "; ".join(f"{k}: {v}" for k, v in answer.items())
        else:
            answer_text = str(answer)

        prompt = f"""You are a policy analyst interpreting a questionnaire response.

Policy Dimension: {dimension.name}
Dimension Description: {dimension.description}
Scale: {dimension.policy_spectrum_description}

Question asked: "{question}"
Person's answer: "{answer_text}"

The person gave a short structured answer. Your job is to determine what their
answer actually means in the context of the question and the policy dimension.

For example:
- "yes" to "Do you support cutting public services?" → low score on public services dimension
- "yes" to "Do you support increasing investment in healthcare?" → high score on healthcare dimension
- "no" to "Do you oppose raising taxes?" → high score on tax/revenue dimension

Instructions:
1. Read the question carefully to understand what a positive answer means
2. Determine whether a "{answer_text}" answer to THIS question indicates
   support or opposition on the "{dimension.name}" dimension
3. Assign a position score on a 0-100 scale where: {dimension.policy_spectrum_description}
4. Rate your confidence (0.0-1.0) in this interpretation

Return ONLY valid JSON with no extra text:
{{
    "position_score": 72.0,
    "confidence": 0.90,
    "reasoning": "A 'yes' to this question indicates support for X, placing them at the higher end of the {dimension.name} scale."
}}"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a policy analyst. Interpret questionnaire answers in context. "
                    "Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await llm_service.call_llm(
                messages,
                max_tokens=matching_config.position_llm_max_tokens,
                temperature=matching_config.position_inference_temperature,
            )
            analysis = llm_service._extract_json_from_response(response)

            if analysis and "position_score" in analysis:
                score = float(analysis["position_score"])
                # Sanity check — score must be in valid range
                if 0.0 <= score <= 100.0:
                    self.logger.debug(
                        f"Context-aware interpretation: '{answer_text}' to "
                        f"'{question[:60]}...' → score={score:.1f} "
                        f"(was {basic_position['score']:.1f})"
                    )
                    return {
                        "score": score,
                        "method": "context_aware",
                        "llm_reasoning": analysis.get("reasoning", ""),
                    }

        except Exception as e:
            self.logger.warning(
                f"Context-aware interpretation failed, falling back to basic: {str(e)}"
            )

        # Fallback to basic position if LLM call fails
        return basic_position

    @staticmethod
    def _is_policy_relevant(answer: Any) -> bool:
        """
        Returns True if the answer contains enough substance to infer a
        policy position from.  Filters out things like names, bare ages,
        single-word filler responses, etc.

        Booleans and structured types (list/dict) are always considered relevant
        because they come from structured questionnaire widgets.
        Text answers must meet a minimum length threshold.
        """
        if isinstance(answer, bool):
            return True
        if isinstance(answer, (list, dict)):
            return True
        if isinstance(answer, str):
            stripped = answer.strip()
            # Must be long enough to carry policy meaning
            if len(stripped) < matching_config.min_answer_length_for_text:
                return False
            # Reject pure numeric answers (e.g. age "32")
            if stripped.isdigit():
                return False
            return True
        return False

    def _extract_basic_position(self, answer: Any, dimension: PolicyDimension) -> Dict[str, float]:
        """
        Extract basic position score from the direct answer.
        Non-policy-relevant answers return None so the caller can skip them.
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

            # Free-text — only proceed if policy-relevant
            if not self._is_policy_relevant(answer):
                # Return a sentinel that callers treat as "skip this response"
                return {"score": -1.0, "method": "irrelevant"}

            return self._analyze_text_sentiment(answer_lower)

        # Structured types (list/dict) — multi-select or ranked answers.
        # Return a temporary mid-point; the context-aware path will refine this.
        if isinstance(answer, (list, dict)):
            return {"score": 65.0, "method": "structured_needs_context"}

        # Unknown type — skip
        return {"score": -1.0, "method": "irrelevant"}

    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Basic sentiment analysis for free-text responses.
        Uses a small set of strong policy indicator words.
        """
        positive_indicators = len([
            w for w in ["good", "great", "excellent", "important",
                        "necessary", "should", "must", "support", "invest"]
            if w in text
        ])
        negative_indicators = len([
            w for w in ["bad", "poor", "unnecessary", "shouldn't",
                        "cannot", "won't", "refuse", "oppose", "against"]
            if w in text
        ])

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
                max_tokens=matching_config.position_llm_max_tokens,
                temperature=matching_config.position_inference_temperature,
            )

            analysis = llm_service._extract_json_from_response(response)

            if analysis and "position_score" in analysis:
                # Blend with basic position (weighted average)
                enhanced_score = (
                    basic_position["score"] * matching_config.position_basic_weight
                    + analysis["position_score"] * matching_config.position_llm_weight
                )
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

        base_confidence = matching_config.position_base_confidence

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
        if comment and len(comment.strip()) > matching_config.position_min_comment_length:
            base_confidence += 0.1  # More context = higher confidence

        # Adjust based on dimension match quality
        if dimension.confidence < 0.7:
            base_confidence -= 0.1  # Less confident dimension mapping

        # Candidate vs voter adjustment
        if person_type == "candidate" and comment:
            base_confidence += 0.05  # Candidates usually give more detailed responses

        return max(0.1, min(1.0, base_confidence))

    def _extract_reasoning(
        self, answer: Any, comment: str, enhanced_position: Dict[str, float] = None
    ) -> str:
        """
        Extract human-readable reasoning for the position.

        Priority order:
          1. LLM reasoning from context-aware or comment-enhanced interpretation
          2. First sentence of a written comment
          3. Truncated raw answer text
          4. Generic fallback
        """
        # Best case: LLM already produced a reasoning string
        if enhanced_position and enhanced_position.get("llm_reasoning"):
            reasoning = enhanced_position["llm_reasoning"].strip()
            if reasoning:
                return reasoning

        # Written comment — use first sentence
        if comment and len(comment.strip()) > 10:
            sentences = comment.split(".")
            if sentences and sentences[0].strip():
                return sentences[0].strip() + "."

        # Short structured answer — describe it plainly
        if isinstance(answer, bool):
            return "Answered yes" if answer else "Answered no"

        if isinstance(answer, str):
            stripped = answer.strip()
            if len(stripped) <= 50:
                return f'Answered: "{stripped}"'
            return f'"{stripped[:47]}..."'

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
