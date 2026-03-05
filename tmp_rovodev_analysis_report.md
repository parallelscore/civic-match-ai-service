# Detailed Analysis of Three Matching Services

## 1. MATCHING_ENGINE_SERVICE.py

### Purpose
Main orchestrator for multi-layer policy-based voter-candidate matching. Processes voter submissions and returns comprehensive match results against all candidates.

### Key Architecture
- **Main Entry Point**: `process_voter_submission()` - async method that orchestrates the entire matching pipeline
- **Returns**: `MatchResultsResponseSchema` containing ALL matched candidates (not limited to top 3)

### Full Logic Flow

#### Step 1: Candidate Retrieval & Classification (Lines 35-52)
- Fetches all candidates for election
- Separates into **eligible** (have complete profiles) and **ineligible** (incomplete data)
- Logs counts of both categories

#### Step 2: Policy Dimension Discovery (Lines 54-60)
- Extracts all unique questions from voter and all eligible candidates
- Discovers election-wide policy dimensions using `policy_dimension_discovery_service`
- Returns `election_analysis` with discovered dimensions and question mappings

#### Step 3: Voter Policy Profile Creation (Lines 62-63)
- Creates `PersonPolicyProfile` for the voter containing:
  - Policy positions mapped to discovered dimensions
  - Handles **primary dimensions** (main mapping for each question)
  - Handles **secondary dimensions** (weighted at 0.3 default, line 178)
  - Applies consistency analysis

#### Step 4: Eligible Candidate Processing (Lines 65-77)
- Creates policy profiles for all eligible candidates
- For each candidate, calls `enhanced_matching_calculator_service.calculate_enhanced_match()`
- Converts results to legacy format for response

#### Step 5: Ineligible Candidates (Lines 79-83)
- Creates 0% match results with category "UNMATCH"
- Explanation: "This candidate has not completed their profile and/or questionnaire, so no policy comparison could be made."

#### Step 6: Sorting & Categorization (Lines 85-89)
- Sorts ALL matches by match percentage (highest first)
- Calls `_assign_match_categories()` to categorize:
  - **TOP**: First 3 calculated matches (>0%)
  - **OTHER**: Remaining calculated matches (>0%)
  - **UNMATCH**: All 0% matches

#### Step 7: Voter Values Profile Generation (Lines 94-95)
- Uses `_generate_voter_values_profile()` to create list of voter's core values
- Returns up to 6 items sorted by priority (High → Medium → Low)

#### Step 8: Processing Quality Determination (Lines 97-100)
- Determines `processing_method` and `confidence_score`
- Methods: "policy_enhanced", "policy_based", "basic_policy", or "fallback"

### Helper Methods & Logic

#### `_extract_all_questions()` (Lines 118-133)
- Combines unique questions from voter responses and all candidate responses
- Returns as list for dimension discovery

#### `_create_voter_policy_profile()` (Lines 135-200)
- **Critical Logic**: Maps voter responses to discovered dimensions
- For each voter response:
  1. Finds question-to-dimension mapping
  2. Finds primary dimension
  3. Calls `position_inference_service.infer_policy_position()` to get position
  4. For secondary dimensions (lines 171-189):
     - Default secondary weight: **0.3** (line 178)
     - Multiplies both confidence and intensity_multiplier by this weight
- Analyzes overall consistency of positions
- Returns voter profile with all positions and tensions

#### `_create_candidate_policy_profiles()` (Lines 202-211)
- Iterates through eligible candidates
- Calls `_create_single_candidate_profile()` for each

#### `_create_single_candidate_profile()` (Lines 213-276)
- **Parallel Logic** to voter profile creation
- Includes optional candidate comments (line 242)
- Same secondary dimension weighting (0.3 default)
- Returns candidate profile with consistency analysis

#### `_convert_to_legacy_format()` (Lines 278-307)
- Converts `EnhancedMatchResult` to `CandidateMatchSchema`
- Dimension matches → Issue matches
- Limits **top_aligned_issues to 3** (line 297)
- Sets initial match_category to "PENDING" (will be assigned later)

#### `_generate_voter_values_profile()` (Lines 309-367)
- **Filter Logic**: Only includes dimensions where voter has strong positions
- Threshold: **> 15 points from neutral** (abs(avg_position - 50) > 15, line 341)
- Priority assignment (lines 344-349):
  - **High**: intensity ≥ 1.5 AND deviation > 25
  - **Medium**: intensity ≥ 1.0 OR deviation > 20
  - **Low**: Everything else (that passes filter)
- Calls `_generate_llm_voter_value_description()` for each dimension
- Returns up to **6 items** sorted by priority

#### `_generate_llm_voter_value_description()` (Lines 369-479)
- **Caching**: Checks `cache_service` first
- **LLM Prompt** (lines 418-443):
  - Requires "You" second-person narrative
  - 1-2 sentences max
  - Focus on motivations/values, not just positions
  - System message: "expert at understanding voter motivations"
  - Temperature: **0.4** (line 456)
  - Max tokens: **100** (line 455)
- **Stance Mapping** (lines 407-416):
  - ≥ 75: "strongly support"
  - ≥ 60: "support"
  - ≥ 40: "have mixed feelings about"
  - ≥ 25: "have concerns about"
  - < 25: "oppose"
- **Cleanup** (lines 460-468): Ensures starts with "You" and ends with period
- **Fallback**: `_create_fallback_voter_value_description()` if LLM fails

#### `_assign_match_categories()` (Lines 481-513)
- Separates calculated (>0%) and zero (=0%) matches
- Top 3 calculated get "TOP"
- Remaining calculated get "OTHER"
- All zero-% get "UNMATCH"
- Logs categorization summary

#### `_determine_processing_quality()` (Lines 541-578)
- **Base Confidence**: From avg dimension discovery confidence
- **Adjustments**:
  - Eligible ratio boost: up to 10% (line 569)
  - Min consistency score boost: +0.2 or +0.1 based on quality (lines 572, 574)
- **Output Methods**:
  - "policy_enhanced": 95% max (line 572)
  - "policy_based": 85% max (line 574)
  - "basic_policy": 60% min (line 576)
  - "fallback": 30% fixed (line 578)

#### `_get_alignment_level()` (Lines 628-637)
- Converts alignment score to descriptive text:
  - ≥ 0.8: "Strongly Aligned"
  - ≥ 0.6: "Moderately Aligned"
  - ≥ 0.4: "Somewhat Aligned"
  - < 0.4: "Weakly Aligned"

### Hardcoded Values

| Value | Location | Purpose |
|-------|----------|---------|
| 0.3 | Lines 178, 255 | Secondary dimension default weight |
| 15 | Line 341 | Threshold for "meaningful position" |
| 25 | Line 344 | High priority position deviation threshold |
| 20 | Line 346 | Medium priority threshold |
| 6 | Line 367 | Max voter values profile items returned |
| 3 | Line 297 | Max top_aligned_issues in legacy format |
| 0.4 | Line 456 | LLM temperature for voter value descriptions |
| 100 | Line 455 | LLM max tokens for voter value descriptions |
| 0.95 | Line 572 | Max confidence for "policy_enhanced" method |
| 0.85 | Line 574 | Max confidence for "policy_based" method |
| 0.6 | Line 576 | Min confidence for "basic_policy" method |
| 0.3 | Line 578 | Fixed confidence for "fallback" method |
| 0.1 | Line 569 | Eligible ratio adjustment factor |

### Concerns & Issues

1. **Secondary Dimension Weighting (0.3) is Questionable**
   - Hard-coded default weight for secondary dimensions seems arbitrary
   - No configuration mechanism to adjust per-election
   - Could significantly impact match calculations if secondary mappings are common

2. **Position Strength Threshold (15 points)**
   - Filters out voters with moderate positions
   - May exclude relevant voter values for candidates with different priorities

3. **Hardcoded Limits**
   - 6-item limit on voter values profile could truncate important values
   - 3-item limit on top_aligned_issues in legacy format loses detail

4. **LLM Dependency**
   - Voter value descriptions entirely LLM-dependent
   - Cache key includes hash of reasoning (line 381) - could cause cache misses
   - Temperature 0.4 is relatively deterministic but may not capture variance

5. **Error Handling**
   - Fallback voter value description (line 479) may produce awkward phrasing
   - No maximum length enforcement on fallback descriptions

6. **Consistency Penalty Application**
   - Applied in `enhanced_matching_calculator_service`, not here
   - Creates distributed responsibility for match score calculation

---

## 2. SEMANTIC_MATCHING_SERVICE.py

### Purpose
Provides semantic similarity matching using sentence transformers and embeddings. Finds similar questions between voters and candidates.

### Key Architecture
- Uses `SentenceTransformer` from sentence_transformers library
- Loads model from `settings.EMBEDDING_MODEL` at initialization
- Creates global singleton instance `semantic_service`

### Full Logic

#### Initialization & Model Loading (Lines 13-25)
- Loads embedding model on service instantiation
- Catches exceptions during model load
- Sets model to `None` if loading fails
- Logs success or failure

#### `encode_questions()` (Lines 27-38)
- Takes list of question strings
- Returns `Optional[np.ndarray]` embeddings
- Checks: model exists AND questions non-empty
- Uses `SentenceTransformer.encode()` with `convert_to_numpy=True`
- Logs debug on success, error on failure
- Returns `None` on failure

#### `find_similar_questions()` (Lines 40-85)
- **Purpose**: Find semantically similar questions for ONE voter question
- **Inputs**: 
  - `voter_question`: single question string
  - `candidate_questions`: list of candidate question strings
  - `threshold`: optional similarity threshold (defaults to `settings.EMBEDDING_SIMILARITY_THRESHOLD`)
- **Process**:
  1. Validates model exists and has candidate questions
  2. Gets threshold from settings if not provided
  3. Combines voter + candidate questions into single list (line 52)
  4. Encodes all questions
  5. Extracts voter embedding (first row, line 59)
  6. Extracts candidate embeddings (remaining rows, line 60)
  7. Calculates cosine similarity: voter vs all candidates (line 62)
- **Filtering** (lines 66-75):
  - Iterates through similarity scores
  - Creates `QuestionSimilaritySchema` for matches **≥ threshold**
  - Includes: voter_question, candidate_question, similarity_score, method="embedding", explanation
- **Sorting**: Results sorted by similarity score descending (line 78)
- **Returns**: List of `QuestionSimilaritySchema` objects

#### `batch_find_similar_questions()` (Lines 87-134)
- **Purpose**: Efficiently find similar questions for MULTIPLE voter questions
- **Inputs**:
  - `voter_questions`: list of voter question strings
  - `candidate_questions`: list of candidate question strings
  - `threshold`: optional threshold
- **Process**:
  1. Validates inputs (model, questions exist)
  2. Gets threshold from settings if not provided
  3. Encodes voter questions separately (line 100)
  4. Encodes candidate questions separately (line 101)
  5. Calculates full similarity matrix: voter_embeddings × candidate_embeddings (line 107)
  6. Iterates through each voter question row (lines 110-127):
     - Extracts similarity scores for that voter's row
     - Creates schema objects for matches ≥ threshold
     - Sorts by similarity descending
     - Stores in results dict with voter_question as key
- **Returns**: Dict[voter_question → List[QuestionSimilaritySchema]]

#### `get_question_clusters()` (Lines 136-172)
- **Purpose**: Group questions by semantic similarity using clustering
- **Inputs**:
  - `questions`: list of question strings
  - `num_clusters`: optional number of clusters
- **Validation** (lines 141-142):
  - Requires model loaded
  - Requires at least 2 questions
- **Cluster Count Determination** (lines 153-154):
  - If not provided: `min(max(2, len(questions) // 3), 8)`
  - Minimum 2 clusters
  - Maximum 8 clusters
  - Default: 1 per 3 questions
- **Clustering** (lines 157-158):
  - Uses `KMeans` from sklearn
  - Parameters: `n_clusters`, `random_state=42`, `n_init=10`
  - Fixed random seed ensures reproducibility
- **Grouping** (lines 160-165):
  - Groups questions by cluster label
  - Returns dict: {cluster_label → [questions]}
- **Returns**: Dict[int, List[str]]

### Hardcoded Values

| Value | Location | Purpose |
|-------|----------|---------|
| `settings.EMBEDDING_MODEL` | Line 21 | Model to load (from config) |
| `settings.EMBEDDING_SIMILARITY_THRESHOLD` | Lines 48, 96 | Default similarity threshold |
| 42 | Line 157 | KMeans random_state seed |
| 10 | Line 157 | KMeans n_init parameter |
| 3 | Line 154 | Divisor for default cluster count |
| 2 | Line 154 | Minimum cluster count |
| 8 | Line 154 | Maximum cluster count |

### Concerns & Issues

1. **Model Loading Failure Handling**
   - If model fails to load, all subsequent calls return empty results
   - No retry mechanism or alternative fallback model
   - Silent degradation - app continues without semantic matching capability

2. **Threshold Dependency**
   - Heavily relies on `settings.EMBEDDING_SIMILARITY_THRESHOLD`
   - No validation that threshold is within [0, 1] range
   - If threshold is too high, may return no matches even for very similar questions

3. **Clustering Parameters**
   - Hard-coded KMeans parameters (`random_state=42`, `n_init=10`) are arbitrary
   - `n_init=10` may be too low for reproducibility with sklearn >= 1.2
   - No option to customize clustering algorithm

4. **Memory Efficiency**
   - `find_similar_questions()` creates embedding for single voter question every call
   - No caching of question embeddings
   - Batch method is more efficient but single method less so

5. **Cosine Similarity Range**
   - Assumes cosine similarity output is [0, 1] range
   - SentenceTransformer typically produces [-1, 1] range
   - May not handle negative similarities correctly

6. **Empty Input Handling**
   - Returns empty list/dict if any condition fails
   - No distinction between "no matches" vs "error occurred"

---

## 3. ENHANCED_MATCHING_CALCULATOR_SERVICE.py

### Purpose
Calculates detailed policy match percentages between voter and candidate profiles using dimension-by-dimension comparison.

### Key Architecture
- Compares voter vs candidate `PersonPolicyProfile` objects
- Returns `EnhancedMatchResult` with comprehensive match details
- Uses confidence weighting and consistency analysis

### Full Logic Flow

#### `calculate_enhanced_match()` (Lines 17-76)
- **Main entry point** - orchestrates full match calculation
- **Steps**:
  1. Calculate dimension-by-dimension matches (line 29)
  2. Calculate base match percentage from dimensions (line 35)
  3. Calculate consistency penalty (lines 38-40)
  4. Apply consistency penalty to base (lines 43-46)
  5. Calculate confidence-weighted percentage (line 49)
  6. Generate explanations (lines 52-57)
  7. Identify top aligned dimensions (line 60)
  8. Create and return `EnhancedMatchResult` (lines 63-72)

#### `_calculate_dimension_matches()` (Lines 78-135)
- **Purpose**: Find matches for each common policy dimension
- **Process**:
  1. Creates lookup dicts for voter and candidate positions (lines 88-89)
  2. Finds **common dimensions** (intersection of both sets, line 94)
  3. For each common dimension (lines 96-131):
     - Gets voter and candidate positions
     - Calculates raw alignment score (line 101)
     - Applies confidence weighting (lines 104-108)
     - Generates LLM alignment explanation (lines 112-117)
     - Creates `DimensionMatch` object with both descriptions (lines 119-129)
- **Returns**: List of `DimensionMatch` objects (only for common dimensions)

#### `_calculate_position_alignment()` (Lines 137-158)
- **Core Scoring Logic**:
  1. **Position Distance** (line 143): `abs(voter_score - candidate_score)`
  2. **Base Similarity** (line 146): `max(0, (100 - distance) / 100)`
     - 0 distance = 1.0 similarity
     - 100+ distance = 0 similarity
  3. **Intensity Weighting** (lines 149-150):
     - Voter intensity multiplier normalized: `(intensity + 1.0) / 3.0`
     - Range: [0.33, 1.0] for typical intensities [0, 2]
     - Weighted similarity = `base_similarity × intensity_weight`
  4. **High Agreement Boost** (lines 153-154):
     - If base_similarity > 0.8: multiply weighted by 1.1 (max 1.0)
     - 10% boost for strong agreements
- **Returns**: Float [0.0, 1.0] alignment score

#### `_apply_confidence_weighting()` (Lines 160-174)
- **Purpose**: Reduce alignment score based on confidence in positions
- **Logic**:
  - Checks if weighting enabled in `matching_config.confidence_weighting_enabled`
  - Uses **minimum confidence** as limiting factor (line 169)
  - Multiplies alignment score by combined confidence (line 172)
- **Example**: 0.8 alignment × 0.7 confidence = 0.56 weighted score
- **Returns**: Float weighted alignment score

#### `_calculate_base_match_percentage()` (Lines 176-201)
- **Purpose**: Convert dimension matches to overall match percentage
- **Logic**:
  1. **Intensity Weighting** (lines 185-193):
     - Each dimension weighted by voter's intensity for that dimension
     - Weighted score = `alignment_score × voter_intensity`
     - Total weight = sum of all intensities
  2. **Weighted Average** (lines 196-199):
     - If total_weight > 0: `(total_weighted_score / total_weight) × 100`
     - Fallback: simple average if weights are 0
  3. **Bounds** (line 201): Clamps to [0, 100]
- **Example**: 
  - 2 dimensions: alignment [0.7, 0.9], intensities [1.5, 1.0]
  - Weighted scores: [1.05, 0.9], total_weight: 2.5
  - Base % = (1.95 / 2.5) × 100 = 78%
- **Returns**: Float [0, 100]

#### `_calculate_confidence_weighted_percentage()` (Lines 203-233)
- **Purpose**: Match percentage weighted by both intensity AND confidence
- **Logic**:
  1. Checks if weighting enabled
  2. If disabled: returns base match percentage (line 212)
  3. For each dimension (lines 218-225):
     - Uses `confidence_weighted_score` from dimension match
     - Multiplies by voter intensity
     - Adds to total weighted score
  4. Weighted average calculation: `(total / total_weight) × 100`
  5. Bounds [0, 100]
- **Returns**: Float [0, 100]

#### `_calculate_consistency_penalty()` (Lines 235-249)
- **Purpose**: Penalize match if either voter or candidate has logical tensions
- **Logic**:
  1. Checks if enabled: `matching_config.enable_consistency_analysis`
  2. Uses **minimum consistency score** (line 244)
  3. Penalty formula (line 247): `consistency_penalty_rate × (1.0 - min_consistency)`
  4. From config: `consistency_penalty_rate` (default unknown, needs config check)
- **Example**: 
  - Voter consistency: 0.9, Candidate: 0.7
  - Min: 0.7, penalty_rate: 0.3
  - Penalty: 0.3 × (1 - 0.7) = 0.09 (9 percentage points)
- **Returns**: Float penalty amount

#### `_generate_alignment_explanation()` (Lines 251-298)
- **Purpose**: Create human-readable explanation + position descriptions
- **Strength Classification** (lines 264-271):
  - ≥ 0.8: "Strong"
  - ≥ 0.6: "Good"
  - ≥ 0.4: "Moderate"
  - < 0.4: "Weak"
- **Position Descriptions** (lines 275-276):
  - Calls `_generate_llm_position_description()` for voter
  - Calls `_generate_llm_position_description()` for candidate
- **Alignment Explanation** (lines 279-284):
  - Position diff ≤ 15: "Both of you share similar views"
  - Position diff ≤ 30: "Somewhat different approaches but generally align"
  - Position diff > 30: "Different perspectives"
  - Prepended with strength classification
- **Low Confidence Note** (lines 294-296):
  - If avg confidence < 0.6: append "(assessment has limited confidence...)"
- **Returns**: Tuple of (explanation, voter_description, candidate_description)

#### `_create_detailed_position_description()` (Lines 300-345)
- **Score-Based Stance** (lines 309-320):
  - ≥ 85: "very strong support"
  - ≥ 70: "strong support"
  - ≥ 60: "moderate support"
  - 40-59: "mixed views"
  - 25-39: "moderate opposition"
  - < 25: "strong opposition"
- **Reasoning Theme Detection** (lines 325-341):
  - Searches reasoning text for keywords:
    - "access", "opportunity", "program" → "focusing on access and opportunities"
    - "funding", "resource", "budget" → "emphasizing funding and resources"
    - "safety", "security", "protection" → "prioritizing safety and security"
    - "mental health", "counseling", "support" → "emphasizing mental health..."
    - "community", "local", "resident" → "focusing on community involvement"
    - "quality", "improvement", "standard" → "emphasizing quality and standards"
    - Fallback: "with specific implementation preferences"
- **Returns**: String description

#### `_describe_position_strength()` (Lines 347-361)
- **Simple Score-to-Text Mapping**:
  - ≥ 80: "Strong support"
  - ≥ 65: "Moderate support"
  - 35-64: "Mixed/neutral"
  - 20-34: "Moderate opposition"
  - < 20: "Strong opposition"
- **Returns**: String

#### `_generate_match_explanation()` (Lines 363-408)
- **Overall Assessment** (lines 378-387):
  - ≥ 80%: "Excellent alignment"
  - ≥ 70%: "Strong alignment"
  - ≥ 60%: "Good alignment"
  - ≥ 45%: "Moderate alignment"
  - < 45%: "Limited alignment"
- **Explanation Components** (lines 389-407):
  1. Overall assessment with dimension count
  2. List top 3 strong matches (≥ 0.7 alignment) if they exist
  3. Note consistency penalty if > 0.1
  4. Note low confidence if > 50% of dimensions have low confidence
- **Returns**: String explanation

#### `_identify_top_aligned_dimensions()` (Lines 410-423)
- **Process**:
  1. Sorts dimension matches by alignment_score descending
  2. Takes top 3 (line 419)
  3. Filters to only include alignment_score ≥ 0.5 (line 420)
  4. Returns dimension names
- **Returns**: List[str] of dimension names (max 3)

#### `_generate_llm_position_description()` (Lines 425-501)
- **Caching** (lines 436-439):
  - Cache key: `position_desc:{dimension}:{score}:{hash(reasoning)}`
  - Returns cached if exists
- **Score Context** (line 442): Calls `_get_score_context()`
- **LLM Prompt** (lines 445-467):
  - Requires max 2 short sentences
  - Sound natural, capture both stance AND reasoning
  - Use "you" for voters, "they" for candidates
  - System: "expert at translating policy positions into natural language"
  - Temperature: **0.3** (line 480) - very deterministic
  - Max tokens: **100** (line 479)
- **Response Cleanup** (lines 484-491):
  - Strips quotes
  - If > 200 chars: truncates to first 2 sentences
- **Cache TTL**: 3600 seconds (1 hour, line 494)
- **Fallback**: `_create_fallback_position_description()` on LLM failure
- **Returns**: String description

#### `_get_score_context()` (Lines 503-516)
- Simple score-to-text mapper (same as `_describe_position_strength`)
- **Returns**: String contextual description

#### `_create_fallback_position_description()` (Lines 518-526)
- **Fallback when LLM fails**:
  - Selects pronoun based on person_type ("You" vs "They")
  - Gets score context
  - If reasoning exists: `"{Pronoun} show {score_desc} based on your belief that {reasoning[:50]}..."`
  - Else: `"{Pronoun} show {score_desc} for this policy area."`
- **Returns**: String

### Hardcoded Values

| Value | Location | Purpose |
|-------|----------|---------|
| 0.33-1.0 | Line 149 | Intensity weight range |
| 1.1 | Line 154 | High agreement boost multiplier |
| 0.8 | Line 153 | Threshold for high agreement boost |
| 0.33 | Line 149 | Min intensity weight (intensity=0) |
| 0.78-1.0 | Line 149 | Typical intensity weight range |
| 0.8, 0.6, 0.4 | Lines 264-271 | Strength classification thresholds |
| 15, 30 | Lines 279-284 | Position diff thresholds for explanation |
| 0.6 | Line 295 | Low confidence threshold for note |
| 80, 70, 60, 45 | Lines 378-387 | Overall assessment percentage thresholds |
| 0.7 | Line 394 | Strong matches threshold |
| 3 | Line 419 | Max top aligned dimensions |
| 0.5 | Line 420 | Min alignment for top dimensions |
| 0.1 | Line 400 | Consistency penalty note threshold |
| 0.5 | Line 405 | Low confidence prevalence threshold |
| 85, 70, 60, 40, 25 | Lines 309-320 | Stance classification score thresholds |
| 85, 70, 60, 35, 20 | Lines 352-361 | Strength description score thresholds |
| 80, 65, 35, 20 | Lines 505-516 | Score context thresholds |
| 0.3 | Line 480 | LLM temperature for position descriptions |
| 100 | Line 479 | LLM max tokens for position descriptions |
| 200 | Line 487 | Max length for position descriptions |
| 3600 | Line 494 | Cache TTL for position descriptions (seconds) |
| 42 | Line 157 (semantic) | KMeans random seed |
| "position_desc" | Line 436 | Cache key prefix |
| 50 | Line 524 | Truncation length for fallback reasoning |

### Concerns & Issues

1. **Intensity Weighting Formula (Line 149)**
   - `(intensity + 1.0) / 3.0` is opaque
   - For intensity=0: weight=0.33 (33%)
   - For intensity=2: weight=1.0 (100%)
   - No documentation of why this specific formula
   - Doesn't align well with typical intensity ranges

2. **High Agreement Boost (Line 154)**
   - 10% boost for similarity > 0.8 seems arbitrary
   - Applied AFTER intensity weighting, potentially distorting signal
   - Asymmetric: doesn't penalize low agreement areas

3. **Confidence Weighting Dependency**
   - Heavily relies on `matching_config.confidence_weighting_enabled` flag
   - If disabled, confidence-weighted percentage same as base (line 212)
   - Creates two different calculation paths that may diverge

4. **Missing Common Dimensions Handling**
   - `_calculate_dimension_matches()` only returns matches for **common** dimensions
   - Voters and candidates with different dimension coverage get unfair comparisons
   - No penalty or handling for completely unaligned candidates

5. **LLM Cache Key Fragility (Line 436)**
   - Uses `hash(position.reasoning)` which could collide
   - Different reasoning with same hash would return wrong cached description
   - No hash collision detection

6. **Position Description Theme Detection (Lines 325-341)**
   - Hard-coded keyword list is domain-specific (education)
   - May not work for other election types
   - Keyword matching is brittle (e.g., "school" not in list)

7. **Score-to-Text Thresholds Multiple **
   - Three different threshold sets: detailed (lines 309-320), describe (lines 352-361), context (lines 505-516)
   - Inconsistent boundaries:
     - Detailed: 85/70/60/40/25
     - Describe: 80/65/35/20
     - Context: 85/70/60/40/25
   - Could produce contradictory descriptions

8. **Match Explanation Word Choice (Line 296)**
   - Hard-coded string: "(assessment has limited confidence due to different question types)"
   - Assumes root cause is question type differences
   - May not reflect actual reason for low confidence

9. **Temperature Sensitivity**
   - Position descriptions use temp=0.3 (very deterministic)
   - Voter value descriptions use temp=0.4 (slightly more varied)
   - No justification for different temperatures
