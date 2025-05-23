import json
import re
import asyncio
from typing import List, Dict, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

import openai
import anthropic
from app.core.config import settings
from app.utils.logging_util import setup_logger
from app.schemas.voters_schema import QuestionTopicSchema, QuestionSimilaritySchema


class LLMService:
    """Service for LLM-based topic discovery and question matching."""

    def __init__(self):
        self.logger = setup_logger(__name__)
        self.provider = settings.LLM_PROVIDER

        if self.provider == "openai" and settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        elif self.provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
            self.logger.error(f"No valid API key found for provider: {self.provider}")
            self.client = None

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response, handling various formats."""
        if not response:
            return None

        # Remove any markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        # Try to find JSON in the response
        json_patterns = [
            r'\[.*\]',  # Array
            r'\{.*\}',  # Object
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # If no JSON found, try parsing the entire response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            self.logger.warning(f"Could not extract JSON from LLM response: {response[:200]}...")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_llm(self, messages: List[Dict], max_tokens: int = None) -> str:
        """Make a call to the configured LLM with retry logic."""
        if not self.client:
            raise ValueError("LLM client not properly initialized")

        max_tokens = max_tokens or settings.LLM_MAX_TOKENS

        try:
            if self.provider == "openai":
                response = await self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=settings.LLM_TEMPERATURE,
                    timeout=settings.LLM_TIMEOUT_SECONDS
                )
                return response.choices[0].message.content

            elif self.provider == "anthropic":
                # Convert OpenAI format to Anthropic format
                system_message = None
                user_messages = []

                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        user_messages.append(msg)

                response = await self.client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=max_tokens,
                    temperature=settings.LLM_TEMPERATURE,
                    system=system_message,
                    messages=user_messages,
                    timeout=settings.LLM_TIMEOUT_SECONDS
                )
                return response.content[0].text

        except Exception as e:
            self.logger.error(f"LLM API call failed: {str(e)}")
            raise

    async def discover_election_topics(self, all_questions: List[str], election_context: str = "") -> List[QuestionTopicSchema]:
        """
        Discover topics from all questions in an election using LLM.
        """
        self.logger.info(f"Discovering topics for {len(all_questions)} questions")

        # Limit to first 20 questions to avoid token limits
        questions_subset = all_questions[:20]
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_subset)])

        prompt = f"""
        Analyze the following election questions and identify the main political topics/themes.
        
        Election Context: {election_context or "Local election"}
        
        Questions:
        {questions_text}
        
        Identify 5-8 main topics that these questions cover. For each topic:
        1. Provide a clear topic_id (lowercase, underscore format like "education_access")
        2. Provide a topic_name (title case like "Education Access")  
        3. Provide a topic_description
        4. List the question numbers that belong to this topic
        5. Assign importance_weight between 0.5 and 1.0
        
        IMPORTANT: Return ONLY a valid JSON array, no other text or explanation.
        
        Example format:
        [
            {{
                "topic_id": "education_access",
                "topic_name": "Education Access",
                "topic_description": "Questions about improving access to educational opportunities",
                "question_numbers": [1, 3, 7],
                "importance_weight": 0.9
            }}
        ]
        """

        messages = [
            {"role": "system", "content": "You are a political analyst. Return only valid JSON arrays, no explanations or markdown."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self._call_llm(messages)
            topics_data = self._extract_json_from_response(response)

            if not topics_data:
                self.logger.warning("LLM returned invalid JSON for topic discovery")
                return []

            topics = []
            for topic_data in topics_data:
                # Map question numbers back to actual questions
                question_numbers = topic_data.get("question_numbers", [])
                topic_questions = [questions_subset[i-1] for i in question_numbers if 0 < i <= len(questions_subset)]

                topic = QuestionTopicSchema(
                    topic_id=topic_data.get("topic_id", "unknown"),
                    topic_name=topic_data.get("topic_name", "Unknown Topic"),
                    topic_description=topic_data.get("topic_description", ""),
                    questions=topic_questions,
                    importance_weight=topic_data.get("importance_weight", 1.0)
                )
                topics.append(topic)

            self.logger.info(f"Discovered {len(topics)} topics")
            return topics

        except Exception as e:
            self.logger.error(f"Topic discovery failed: {str(e)}")
            return []

    async def find_similar_questions(self, voter_question: str, candidate_questions: List[str],
                                     topic_context: str = "") -> List[QuestionSimilaritySchema]:
        """
        Find candidate questions that are semantically similar to a voter question.
        """
        if not candidate_questions:
            return []

        # Limit candidate questions to avoid token limits
        candidate_questions_subset = candidate_questions[:15]
        candidate_questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(candidate_questions_subset)])

        prompt = f"""
        Find candidate questions similar to the voter question.
        
        Voter Question: "{voter_question}"
        
        Candidate Questions:
        {candidate_questions_text}
        
        Rate similarity from 0.0 to 1.0 (only include if >= 0.6).
        
        Return ONLY a JSON array:
        [
            {{
                "candidate_question_number": 2,
                "similarity_score": 0.85,
                "explanation": "Both ask about mental health support"
            }}
        ]
        """

        messages = [
            {"role": "system", "content": "You are an expert at semantic analysis. Return only valid JSON arrays."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self._call_llm(messages)
            similarities_data = self._extract_json_from_response(response)

            if not similarities_data:
                return []

            similarities = []
            for sim_data in similarities_data:
                question_num = sim_data.get("candidate_question_number", 0)
                if 1 <= question_num <= len(candidate_questions_subset):
                    similarity = QuestionSimilaritySchema(
                        voter_question=voter_question,
                        candidate_question=candidate_questions_subset[question_num - 1],
                        similarity_score=sim_data.get("similarity_score", 0.0),
                        similarity_method="llm",
                        explanation=sim_data.get("explanation", "")
                    )
                    similarities.append(similarity)

            return similarities

        except Exception as e:
            self.logger.error(f"Question similarity analysis failed: {str(e)}")
            return []

    async def analyze_position_alignment(self, voter_answer: str, candidate_answer: str,
                                         question_context: str) -> Tuple[float, str]:
        """
        Analyze how well voter and candidate positions align on a specific question.
        """
        prompt = f"""
        Rate alignment between these positions (0.0 to 1.0):
        
        Question: {question_context}
        Voter: {voter_answer}
        Candidate: {candidate_answer}
        
        Return ONLY a JSON object:
        {{
            "alignment_score": 0.85,
            "explanation": "Both strongly support the policy"
        }}
        """

        messages = [
            {"role": "system", "content": "You are a policy analyst. Return only valid JSON objects."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self._call_llm(messages)
            alignment_data = self._extract_json_from_response(response)

            if not alignment_data:
                return 0.5, "Could not analyze alignment"

            score = alignment_data.get("alignment_score", 0.0)
            explanation = alignment_data.get("explanation", "")

            return score, explanation

        except Exception as e:
            self.logger.error(f"Position alignment analysis failed: {str(e)}")
            return 0.5, "Analysis failed"

    async def generate_voter_profile(self, voter_responses: List[Dict], discovered_topics: List[QuestionTopicSchema]) -> List[Dict]:
        """
        Generate a voter's political values profile based on their responses.
        """
        responses_text = "\n".join([f"- {r['question']}: {r['answer']}" for r in voter_responses[:8]])  # Limit responses

        prompt = f"""
        Create a political values profile from these responses:
        
        {responses_text}
        
        Create 3-5 key values with High/Medium/Low priority.
        
        Return ONLY a JSON array:
        [
            {{
                "issue": "Education Access",
                "description": "You strongly support expanding educational opportunities",
                "priority_level": "High"
            }}
        ]
        """

        messages = [
            {"role": "system", "content": "You are a political analyst. Return only valid JSON arrays."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self._call_llm(messages)
            profile_data = self._extract_json_from_response(response)
            return profile_data or []

        except Exception as e:
            self.logger.error(f"Voter profile generation failed: {str(e)}")
            return []


# Create global instance
llm_service = LLMService()