# app/services/policy_dimension_discovery_service.py

from datetime import datetime
from typing import List, Dict
from app.services.llm_service import llm_service
from app.services.caching_service import cache_service
from app.schemas.policy_matching_schema import PolicyDimension, QuestionDimensionMapping, ElectionPolicyAnalysis
from app.core.matching_config import matching_config
from app.utils.logging_util import setup_logger

class PolicyDimensionDiscoveryService:
    """Service for discovering policy dimensions from election questions"""

    def __init__(self):
        self.logger = setup_logger(__name__)

    async def discover_election_policy_dimensions(
            self,
            election_id: str,
            all_questions: List[str]
    ) -> ElectionPolicyAnalysis:
        """
        Main entry point for discovering policy dimensions for an election
        """

        # Check cache first
        if matching_config.cache_dimension_discovery:
            cached_analysis = await cache_service.get_election_policy_analysis(election_id)
            if cached_analysis:
                self.logger.info(f"Using cached policy analysis for election {election_id}")
                return ElectionPolicyAnalysis(**cached_analysis)

        self.logger.info(f"Discovering policy dimensions for election {election_id} with {len(all_questions)} questions")

        # Step 1: Discover core policy dimensions
        dimensions = await self._discover_core_dimensions(all_questions)

        # Step 2: Map questions to dimensions
        question_mappings = await self._map_questions_to_dimensions(all_questions, dimensions)

        # Step 3: Validate and refine
        analysis = await self._validate_and_refine_analysis(dimensions, question_mappings)

        # Step 4: Create a final analysis object
        election_analysis = ElectionPolicyAnalysis(
            election_id=election_id,
            discovered_dimensions=analysis["dimensions"],
            question_mappings=analysis["mappings"],
            discovery_confidence=analysis["confidence"],
            analysis_timestamp=str(datetime.now())
        )

        # Cache the results
        if matching_config.cache_dimension_discovery:
            await cache_service.cache_election_policy_analysis(
                election_id,
                election_analysis.model_dump(),
                ttl_seconds=matching_config.dimension_cache_ttl
            )

        self.logger.info(f"Discovered {len(election_analysis.discovered_dimensions)} policy dimensions for election {election_id}")
        return election_analysis

    async def _discover_core_dimensions(self, all_questions: List[str]) -> List[PolicyDimension]:
        """
        Use LLM to discover the core policy dimensions from questions
        """

        # Prepare questions for analysis (limit to avoid token limits)
        sample_limit = matching_config.dimension_discovery_question_sample
        questions_sample = all_questions[:sample_limit] if len(all_questions) > sample_limit else all_questions
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_sample)])

        prompt = f"""
        Analyze these election questions and identify the {matching_config.max_policy_dimensions} most important policy dimensions being debated.
        
        Questions:
        {questions_text}
        
        For each dimension:
        1. Create a dimension_id (lowercase_with_underscores)
        2. Provide a clear name (Title Case)
        3. Write a description of what this dimension covers
        4. Describe the policy spectrum (what 0 means vs what 100 means)
        5. List 3-5 keywords that indicate this dimension
        6. Assign a confidence score (0.0-1.0) for how clearly this dimension emerges from the questions
        
        Focus on dimensions that:
        - Are actually debated in these questions (not generic political topics)
        - Can be measured on a spectrum (support/opposition, more/less, etc.)
        - Are distinct from each other
        - Cover the most important policy areas in these questions
        
        Return ONLY a valid JSON array with exactly {matching_config.max_policy_dimensions} dimensions:
        [
            {{
                "dimension_id": "education_access",
                "name": "Education Access",
                "description": "Policies related to expanding access to educational opportunities and programs",
                "policy_spectrum_description": "Limited Access (0) to Universal Access (100)",
                "keywords": ["education", "school", "access", "programs", "opportunities"],
                "confidence": 0.9
            }}
        ]
        """

        messages = [
            {"role": "system", "content": "You are a policy analyst specializing in election issue analysis. Return only valid JSON arrays."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await llm_service.call_llm(
                messages,
                max_tokens=matching_config.llm_max_tokens_dimension_discovery,
                temperature=matching_config.dimension_discovery_temperature,
            )

            dimensions_data = llm_service._extract_json_from_response(response)

            # Accept anything between the minimum sensible count (3) and the
            # configured maximum.  Requiring an exact count was too brittle —
            # the LLM might return 5 instead of 6 and that is still useful.
            min_acceptable = 3
            received = len(dimensions_data) if dimensions_data else 0
            if not dimensions_data or received < min_acceptable:
                self.logger.warning(
                    f"LLM returned {received} dimensions — below minimum of "
                    f"{min_acceptable}. Falling back to keyword-derived dimensions."
                )
                return await self._create_fallback_dimensions(all_questions)

            if received > matching_config.max_policy_dimensions:
                self.logger.info(
                    f"LLM returned {received} dimensions — trimming to "
                    f"configured maximum of {matching_config.max_policy_dimensions}."
                )
                dimensions_data = dimensions_data[:matching_config.max_policy_dimensions]

            dimensions = []
            for dim_data in dimensions_data:
                dimension = PolicyDimension(**dim_data)
                dimensions.append(dimension)

            return dimensions

        except Exception as e:
            self.logger.error(f"Dimension discovery failed: {str(e)}")
            return await self._create_fallback_dimensions(all_questions)

    async def _map_questions_to_dimensions(
            self,
            all_questions: List[str],
            dimensions: List[PolicyDimension]
    ) -> List[QuestionDimensionMapping]:
        """
        Map each question to policy dimensions
        """

        mappings = []
        dimension_info = "\n".join([
            f"- {dim.dimension_id}: {dim.name} - {dim.description}"
            for dim in dimensions
        ])

        # Process questions in batches to avoid token limits
        batch_size = matching_config.dimension_mapping_batch_size
        for i in range(0, len(all_questions), batch_size):
            batch = all_questions[i:i+batch_size]
            batch_mappings = await self._map_question_batch(batch, dimension_info, dimensions)
            mappings.extend(batch_mappings)

        return mappings

    async def _map_question_batch(
            self,
            questions: List[str],
            dimension_info: str,
            dimensions: List[PolicyDimension]
    ) -> List[QuestionDimensionMapping]:
        """
        Map a batch of questions to dimensions
        """

        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        dimension_ids = [dim.dimension_id for dim in dimensions]

        prompt = f"""
        Map these questions to policy dimensions. Each question should have:
        - One primary dimension (most relevant)
        - Optional secondary dimensions (if relevant)
        - Confidence score for the mapping
        
        Available Dimensions:
        {dimension_info}
        
        Questions to Map:
        {questions_text}
        
        Return ONLY a valid JSON array with one object per question:
        [
            {{
                "question": "exact question text",
                "primary_dimension_id": "dimension_id",
                "secondary_dimension_ids": ["other_dimension_id"],
                "primary_weight": 1.0,
                "secondary_weights": {{"other_dimension_id": 0.3}},
                "mapping_confidence": 0.85
            }}
        ]
        
        Guidelines:
        - primary_weight is always 1.0
        - secondary_weights should be 0.1-0.5 
        - mapping_confidence: 0.9+ for clear mapping, 0.7+ for good mapping, 0.5+ for uncertain
        - Only include secondary dimensions if truly relevant
        """

        messages = [
            {"role": "system", "content": "You are a policy analyst. Map questions to policy dimensions accurately. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await llm_service.call_llm(
                messages,
                max_tokens=matching_config.llm_max_tokens_question_mapping,
                temperature=matching_config.dimension_discovery_temperature,
            )

            mappings_data = llm_service._extract_json_from_response(response)

            if not mappings_data:
                return self._create_fallback_mappings(questions, dimensions)

            mappings = []
            for mapping_data in mappings_data:
                # Validate dimension IDs exist
                if mapping_data.get("primary_dimension_id") in dimension_ids:
                    mapping = QuestionDimensionMapping(**mapping_data)
                    mappings.append(mapping)
                else:
                    # Fallback mapping
                    fallback_mapping = self._create_fallback_mapping(mapping_data["question"], dimensions)
                    mappings.append(fallback_mapping)

            return mappings

        except Exception as e:
            self.logger.error(f"Question mapping failed: {str(e)}")
            return self._create_fallback_mappings(questions, dimensions)

    async def _validate_and_refine_analysis(
            self,
            dimensions: List[PolicyDimension],
            mappings: List[QuestionDimensionMapping]
    ) -> Dict:
        """
        Validate the analysis and refine if necessary
        """

        # Check coverage - ensure all dimensions have at least one question
        dimension_coverage = {}
        for dim in dimensions:
            dimension_coverage[dim.dimension_id] = 0

        for mapping in mappings:
            dimension_coverage[mapping.primary_dimension_id] += 1
            for sec_dim in mapping.secondary_dimension_ids:
                dimension_coverage[sec_dim] = dimension_coverage.get(sec_dim, 0) + 0.5

        # Calculate overall confidence
        mapping_confidences = [m.mapping_confidence for m in mappings]
        dimension_confidences = [d.confidence for d in dimensions]
        overall_confidence = (
                                     sum(mapping_confidences) / len(mapping_confidences) +
                                     sum(dimension_confidences) / len(dimension_confidences)
                             ) / 2

        uncovered_dimensions = [dim_id for dim_id, count in dimension_coverage.items() if count == 0]

        if uncovered_dimensions:
            self.logger.warning(f"Dimensions with no questions: {uncovered_dimensions}")
            # Remove uncovered dimensions and adjust confidence
            dimensions = [d for d in dimensions if d.dimension_id not in uncovered_dimensions]
            overall_confidence *= 0.8

        return {
            "dimensions": dimensions,
            "mappings": mappings,
            "confidence": overall_confidence,
            "coverage": dimension_coverage
        }

    async def _create_fallback_dimensions(self, questions: List[str]) -> List[PolicyDimension]:
        """
        Create fallback dimensions when LLM discovery fails.

        V2: dimensions are derived from the actual questions using keyword
        frequency analysis — fully generic, works for any election domain.
        We build one dimension per broad policy cluster found in the text.
        """
        # Broad policy clusters — keywords that signal a topic area
        policy_clusters = [
            {
                "dimension_id": "spending_and_funding",
                "name": "Spending & Funding",
                "description": "Policies about public spending, budgets, and how programmes are funded",
                "policy_spectrum_description": "Reduce Spending (0) to Increase Spending (100)",
                "keywords": ["fund", "budget", "spend", "invest", "cost", "resource", "finance"],
            },
            {
                "dimension_id": "public_services",
                "name": "Public Services",
                "description": "The level and quality of services provided to the public",
                "policy_spectrum_description": "Minimal Services (0) to Expanded Services (100)",
                "keywords": ["service", "program", "support", "access", "provide", "benefit"],
            },
            {
                "dimension_id": "government_role",
                "name": "Government Role",
                "description": "How much government should be involved in regulating and deciding policy",
                "policy_spectrum_description": "Limited Government (0) to Active Government (100)",
                "keywords": ["government", "regulation", "policy", "mandate", "legislation", "law"],
            },
            {
                "dimension_id": "community_and_local",
                "name": "Community & Local Control",
                "description": "The extent to which local communities control decisions that affect them",
                "policy_spectrum_description": "Centralised Control (0) to Local Autonomy (100)",
                "keywords": ["community", "local", "district", "neighbourhood", "resident", "autonomy"],
            },
            {
                "dimension_id": "safety_and_security",
                "name": "Safety & Security",
                "description": "Approaches to public safety, law enforcement, and security measures",
                "policy_spectrum_description": "Minimal Intervention (0) to High Intervention (100)",
                "keywords": ["safety", "security", "police", "enforcement", "protection", "crime"],
            },
            {
                "dimension_id": "equity_and_inclusion",
                "name": "Equity & Inclusion",
                "description": "Policies that promote equal access and inclusion for all groups",
                "policy_spectrum_description": "Status Quo (0) to Full Equity Focus (100)",
                "keywords": ["equity", "equal", "inclusion", "diversity", "access", "fair", "opportunity"],
            },
        ]

        # Score each cluster by how many of its keywords appear in the questions
        questions_text = " ".join(questions).lower()
        scored = []
        for cluster in policy_clusters:
            score = sum(1 for kw in cluster["keywords"] if kw in questions_text)
            scored.append((score, cluster))

        # Sort by relevance — most relevant clusters first
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build PolicyDimension objects for the top N clusters
        n = matching_config.max_policy_dimensions
        fallback_dimensions = []
        for _, cluster in scored[:n]:
            fallback_dimensions.append(
                PolicyDimension(
                    dimension_id=cluster["dimension_id"],
                    name=cluster["name"],
                    description=cluster["description"],
                    policy_spectrum_description=cluster["policy_spectrum_description"],
                    keywords=cluster["keywords"],
                    confidence=0.5,  # Lower confidence to signal these are fallbacks
                )
            )

        # Guarantee we always return at least one dimension
        if not fallback_dimensions:
            fallback_dimensions.append(
                PolicyDimension(
                    dimension_id="general_policy",
                    name="General Policy",
                    description="Overall policy position inferred from questionnaire responses",
                    policy_spectrum_description="Conservative Approach (0) to Progressive Approach (100)",
                    keywords=["policy", "support", "change", "reform", "improve"],
                    confidence=0.4,
                )
            )

        self.logger.warning(
            f"Using keyword-derived fallback dimensions: "
            f"{[d.dimension_id for d in fallback_dimensions]}"
        )
        return fallback_dimensions

    def _create_fallback_mappings(
            self,
            questions: List[str],
            dimensions: List[PolicyDimension]
    ) -> List[QuestionDimensionMapping]:
        """
        Create fallback question mappings when LLM mapping fails
        """

        mappings = []
        for question in questions:
            # Simple keyword-based mapping
            best_match = self._find_best_dimension_match(question, dimensions)

            mapping = QuestionDimensionMapping(
                question=question,
                primary_dimension_id=best_match.dimension_id,
                secondary_dimension_ids=[],
                primary_weight=1.0,
                secondary_weights={},
                mapping_confidence=0.5
            )
            mappings.append(mapping)

        return mappings

    def _create_fallback_mapping(
            self,
            question: str,
            dimensions: List[PolicyDimension]
    ) -> QuestionDimensionMapping:
        """
        Create fallback mapping for a single question
        """

        best_match = self._find_best_dimension_match(question, dimensions)

        return QuestionDimensionMapping(
            question=question,
            primary_dimension_id=best_match.dimension_id,
            secondary_dimension_ids=[],
            primary_weight=1.0,
            secondary_weights={},
            mapping_confidence=0.4
        )

    def _find_best_dimension_match(
            self,
            question: str,
            dimensions: List[PolicyDimension]
    ) -> PolicyDimension:
        """
        Find the best dimension match using keyword overlap
        """

        question_lower = question.lower()
        best_score = 0
        best_dimension = dimensions[0]  # Default fallback

        for dimension in dimensions:
            score = sum(1 for keyword in dimension.keywords if keyword.lower() in question_lower)
            if score > best_score:
                best_score = score
                best_dimension = dimension

        return best_dimension

# Create a service instance
policy_dimension_discovery_service = PolicyDimensionDiscoveryService()
