# app/services/llm_direct_matching_service.py

import json
from typing import List, Dict
from openai import AsyncOpenAI
from app.core.config import settings
from app.schemas.voter_submission_schema import Response
import logging


class LLMDirectMatchingService:
    """
    Direct LLM-based matching that compares voter and candidate responses
    without requiring dimension discovery or question mapping.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.logger = logging.getLogger(__name__)

    async def calculate_match(
        self,
        voter_responses: List[Response],
        candidate_responses: List[Response],
        voter_id: str,
        candidate_id: str
    ) -> Dict:
        """
        Directly compare voter and candidate responses using LLM.

        Returns:
            {
                "match_percentage": float,  # 0-100
                "common_questions": int,
                "alignment_summary": str,
                "explanation": str,
                "confidence": float
            }
        """

        self.logger.info(
            f"Starting direct LLM match between voter {voter_id} and candidate {candidate_id}"
        )

        # Prepare responses for LLM
        voter_qa = self._format_responses(voter_responses)
        candidate_qa = self._format_responses(candidate_responses)

        # Build prompt
        prompt = self._build_matching_prompt(voter_qa, candidate_qa)

        try:
            # Call OpenAI LLM
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            result = self._parse_llm_response(response.choices[0].message.content)

            self.logger.info(
                f"LLM match result: {result['match_percentage']}% "
                f"({result['common_questions']} common questions)"
            )

            return result

        except Exception as e:
            self.logger.error(f"LLM direct matching failed: {str(e)}")
            # Return conservative fallback
            return {
                "match_percentage": 0.0,
                "common_questions": 0,
                "alignment_summary": "Unable to calculate match due to technical error",
                "explanation": f"Error: {str(e)}",
                "confidence": 0.0
            }

    def _format_responses(self, responses: List[Response]) -> str:
        """Format responses as Q&A pairs for the LLM"""

        formatted = []
        for i, response in enumerate(responses, 1):
            question = response.question
            answer = response.answer
            comment = getattr(response, 'comment', '')

            qa_text = f"{i}. Q: {question}\n   A: {answer}"
            if comment and len(comment.strip()) > 0:
                qa_text += f"\n   Comment: {comment}"

            formatted.append(qa_text)

        return "\n\n".join(formatted)

    def _build_matching_prompt(self, voter_qa: str, candidate_qa: str) -> str:
        """Build the LLM prompt for direct matching"""

        return f"""You are a political alignment analyzer. Compare these voter and candidate responses to determine their policy alignment.

VOTER RESPONSES:
{voter_qa}

CANDIDATE RESPONSES:
{candidate_qa}

YOUR TASK:
1. Identify questions that both voter and candidate answered (overlapping questions)
2. For each overlapping question, assess how closely their positions align
3. Calculate an overall match percentage (0-100%) based on agreement across common topics
4. Provide a brief explanation of alignment

SCORING GUIDELINES:
- 90-100%: Strong alignment on nearly all common issues
- 70-89%: Generally aligned with some differences
- 50-69%: Mixed alignment, agree on some issues, disagree on others
- 30-49%: More disagreement than agreement
- 0-29%: Significant disagreement on most issues

IMPORTANT:
- Only consider questions BOTH answered
- If no overlapping questions exist, return 0% with explanation
- Consider intensity (strongly agree vs agree) in scoring
- Weight substantive comments heavily

Return ONLY a valid JSON object with this structure:
{{
  "match_percentage": <float 0-100>,
  "common_questions": <int count of overlapping questions>,
  "alignment_summary": "<1 sentence summary>",
  "explanation": "<2-3 sentences explaining the match score>",
  "confidence": <float 0-1 indicating confidence in the assessment>
}}"""

    def _parse_llm_response(self, llm_output: str) -> Dict:
        """Parse LLM JSON response"""

        try:
            # Extract JSON from response (handle markdown code blocks)
            llm_output = llm_output.strip()
            if llm_output.startswith("```json"):
                llm_output = llm_output[7:]
            if llm_output.startswith("```"):
                llm_output = llm_output[3:]
            if llm_output.endswith("```"):
                llm_output = llm_output[:-3]

            result = json.loads(llm_output.strip())

            # Validate required fields
            required_fields = ["match_percentage", "common_questions", "alignment_summary", "explanation", "confidence"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Ensure match_percentage is in valid range
            result["match_percentage"] = max(0.0, min(100.0, float(result["match_percentage"])))
            result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}\nRaw output: {llm_output}")
            # Return conservative fallback
            return {
                "match_percentage": 0.0,
                "common_questions": 0,
                "alignment_summary": "Unable to parse alignment analysis",
                "explanation": f"Technical error parsing LLM response: {str(e)}",
                "confidence": 0.0
            }


# Global singleton
llm_direct_matching_service = LLMDirectMatchingService()
