# app/services/llm_direct_matching_service.py

import json
from typing import List, Dict
from openai import AsyncOpenAI
from app.core.config import settings
from app.schemas.voters_schema import VoterResponseItemSchema
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
        voter_responses: List[VoterResponseItemSchema],
        candidate_responses: List[VoterResponseItemSchema],
        voter_id: str,
        candidate_id: str
    ) -> Dict:
        """
        Directly compare voter and candidate responses using LLM with detailed breakdown.

        Returns:
            {
                "match_percentage": float,
                "common_questions": int,
                "alignment_summary": str,
                "explanation": str,
                "confidence": float,
                "issue_matches": [...],
                "top_aligned_issues": [...]
            }
        """

        self.logger.info(
            f"Starting enhanced LLM match between voter {voter_id} and candidate {candidate_id}"
        )

        # Prepare responses for LLM
        voter_qa = self._format_responses(voter_responses)
        candidate_qa = self._format_responses(candidate_responses)

        # Build enhanced matching prompt
        prompt = self._build_enhanced_matching_prompt(voter_qa, candidate_qa)

        try:
            # Call OpenAI LLM with increased tokens for detailed response
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=2000,  # Increased for detailed breakdown
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse enhanced response
            result = self._parse_enhanced_llm_response(response.choices[0].message.content)

            self.logger.info(
                f"Enhanced LLM match result: {result['match_percentage']}% "
                f"({len(result.get('issue_matches', []))} issues analyzed)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Enhanced LLM matching failed: {str(e)}")
            # Return conservative fallback with empty arrays
            return {
                "match_percentage": 0.0,
                "common_questions": 0,
                "alignment_summary": "Unable to calculate match for this candidate at the moment",
                "explanation": f"Technical error occurred during matching: {str(e)}",
                "confidence": 0.0,
                "issue_matches": [],
                "top_aligned_issues": []
            }

    async def extract_voter_values(
        self,
        voter_responses: List[VoterResponseItemSchema]
    ) -> List[Dict]:
        """
        Extract voter's key political values and priorities from their responses.

        Returns:
            [
                {
                    "issue": "Educational Equity",
                    "description": "Strong support for...",
                    "priority_level": "High"
                }
            ]
        """

        voter_qa = self._format_responses(voter_responses)

        prompt = f"""Analyze these voter responses to identify their key political values and priorities.

VOTER RESPONSES:
{voter_qa}

YOUR TASK:
1. Identify 3-5 key political values/priorities reflected in these responses
2. For each value, provide a concise description of the voter's stance
3. Assess priority level based on response intensity and comment depth

Return ONLY a valid JSON object:
{{
  "values": [
    {{
      "issue": "<concise issue name>",
      "description": "<1-2 sentence description of voter's stance>",
      "priority_level": "<High|Medium|Low>"
    }}
  ]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=800,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            result = self._parse_json_response(response.choices[0].message.content)
            return result.get("values", [])

        except Exception as e:
            self.logger.error(f"Voter values extraction failed: {str(e)}")
            return [{
                "issue": "Overall Policy Preferences",
                "description": f"Based on {len(voter_responses)} responses to the questionnaire.",
                "priority_level": "Medium"
            }]

    def _format_responses(self, responses: List[VoterResponseItemSchema]) -> str:
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

    def _build_enhanced_matching_prompt(self, voter_qa: str, candidate_qa: str) -> str:
        """Build enhanced LLM prompt for detailed matching with issue breakdown"""

        return f"""You are a political alignment analyzer. Compare these voter and candidate responses to provide a DETAILED policy alignment analysis.

VOTER RESPONSES:
{voter_qa}

CANDIDATE RESPONSES:
{candidate_qa}

YOUR TASK:
1. Identify ALL overlapping questions (where both answered the same question)
2. For EACH overlapping question:
   - Extract the specific policy issue being addressed
   - Assess alignment strength (Strongly Aligned / Moderately Aligned / Weakly Aligned / Misaligned)
   - Calculate alignment score (0.0 to 1.0)
   - Summarize BOTH positions specifically (not just "they agree")
   - Explain WHY they align or don't (cite specific responses)

3. Identify top 3-5 issues where they align most strongly

4. Calculate overall match percentage (0-100%)

5. Provide a PRACTICAL explanation that:
   - Highlights specific policy agreements (e.g., "Both support raising teacher salaries by at least 15%")
   - Notes meaningful differences (e.g., "Differ on timeline: voter wants immediate action, candidate proposes 3-year plan")
   - Avoids generic statements like "they generally agree on education"

SCORING GUIDELINES:
- 90-100%: Strong alignment on nearly all common issues
- 70-89%: Generally aligned with some differences
- 50-69%: Mixed alignment, agree on some issues, disagree on others
- 30-49%: More disagreement than agreement
- 0-29%: Significant disagreement on most issues

ALIGNMENT LEVELS:
- Strongly Aligned (0.8-1.0): Same position with similar intensity
- Moderately Aligned (0.5-0.79): Same direction but different intensity or nuance
- Weakly Aligned (0.3-0.49): Some overlap but notable differences
- Misaligned (0.0-0.29): Opposing positions or no common ground

Return ONLY a valid JSON object:
{{
  "match_percentage": <float 0-100>,
  "common_questions": <int>,
  "alignment_summary": "<1 sentence overall summary>",
  "explanation": "<2-3 sentences with SPECIFIC policy comparisons, not generic statements>",
  "confidence": <float 0-1>,
  "issue_matches": [
    {{
      "issue": "<specific policy issue name>",
      "alignment": "<Strongly Aligned|Moderately Aligned|Weakly Aligned|Misaligned>",
      "alignment_score": <float 0-1>,
      "voter_position": "<specific voter stance, not just 'agrees'>",
      "candidate_position": "<specific candidate stance, not just 'agrees'>",
      "explanation": "<why they align/don't align on THIS specific issue>"
    }}
  ],
  "top_aligned_issues": ["<issue 1>", "<issue 2>", "<issue 3>"]
}}"""

    def _parse_enhanced_llm_response(self, llm_output: str) -> Dict:
        """Parse enhanced LLM JSON response with issue breakdown"""

        try:
            result = self._parse_json_response(llm_output)

            # Validate required fields
            required_fields = ["match_percentage", "common_questions", "alignment_summary",
                             "explanation", "confidence", "issue_matches", "top_aligned_issues"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Validate match_percentage range
            result["match_percentage"] = max(0.0, min(100.0, float(result["match_percentage"])))
            result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

            # Validate issue_matches structure
            if not isinstance(result["issue_matches"], list):
                raise ValueError("issue_matches must be a list")

            for issue_match in result["issue_matches"]:
                required_issue_fields = ["issue", "alignment", "alignment_score",
                                       "voter_position", "candidate_position", "explanation"]
                for field in required_issue_fields:
                    if field not in issue_match:
                        raise ValueError(f"Missing field {field} in issue_match")

                # Clamp alignment_score
                issue_match["alignment_score"] = max(0.0, min(1.0, float(issue_match["alignment_score"])))

            # Validate top_aligned_issues
            if not isinstance(result["top_aligned_issues"], list):
                raise ValueError("top_aligned_issues must be a list")

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse enhanced LLM response: {str(e)}\nRaw output: {llm_output}")
            # Return conservative fallback
            return {
                "match_percentage": 0.0,
                "common_questions": 0,
                "alignment_summary": "Unable to parse alignment analysis",
                "explanation": f"Technical error parsing LLM response: {str(e)}",
                "confidence": 0.0,
                "issue_matches": [],
                "top_aligned_issues": []
            }

    def _parse_json_response(self, llm_output: str) -> Dict:
        """Parse JSON from LLM response, handling markdown code blocks"""

        llm_output = llm_output.strip()

        # Remove markdown code blocks
        if llm_output.startswith("```json"):
            llm_output = llm_output[7:]
        if llm_output.startswith("```"):
            llm_output = llm_output[3:]
        if llm_output.endswith("```"):
            llm_output = llm_output[:-3]

        return json.loads(llm_output.strip())


# Global singleton
llm_direct_matching_service = LLMDirectMatchingService()
