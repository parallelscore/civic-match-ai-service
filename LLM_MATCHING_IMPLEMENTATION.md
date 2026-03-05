# LLM Direct Matching Implementation

## Overview
Direct LLM-based matching that compares voter and candidate Q&A responses without dimension discovery caching issues.

## Request Structure (UNCHANGED)
```json
{
  "electionId": "uuid",
  "citizenId": "uuid",
  "responses": [
    {
      "questionId": "uuid",
      "question": "Should we increase teacher salaries?",
      "answer": "Strongly Agree",
      "comment": "Teachers deserve better pay"
    }
  ]
}
```

## Response Structure (UNCHANGED)
```json
{
  "citizenId": "uuid",
  "electionId": "uuid",
  "voterValuesProfile": [
    {
      "issue": "Overall Policy Preferences",
      "description": "Based on your 7 responses...",
      "priorityLevel": "Medium"
    }
  ],
  "matches": [
    {
      "candidateId": "uuid",
      "matchPercentage": 75,
      "matchStrengthVisual": 0.75,
      "matchCategory": "TOP",
      "topAlignedIssues": [],
      "issueMatches": [],
      "overallExplanation": "Strong alignment on education funding..."
    }
  ],
  "generatedAt": "2026-03-05T19:00:00",
  "processingMethod": "llm_direct_matching",
  "confidenceScore": 0.85
}
```

## How It Works

### 1. LLM Direct Matching Service
**File**: `app/services/llm_direct_matching_service.py`

- Takes voter Q&A and candidate Q&A
- Sends to OpenAI GPT-4o with structured prompt
- LLM sees full context: questions + answers + comments
- Returns JSON with match percentage, explanation, confidence

**LLM Prompt Structure**:
```
VOTER RESPONSES:
1. Q: Should we increase teacher salaries?
   A: Strongly Agree
   Comment: Teachers deserve better

CANDIDATE RESPONSES:
1. Q: Should we increase teacher salaries?
   A: Agree
2. Q: Should we modernize facilities?
   A: Strongly Agree
...

YOUR TASK:
- Identify overlapping questions
- Assess alignment on each
- Calculate match percentage (0-100%)
- Provide explanation

Return JSON:
{
  "match_percentage": 75.0,
  "common_questions": 7,
  "alignment_summary": "Strong alignment on education",
  "explanation": "Voter and candidate agree on 6 of 7...",
  "confidence": 0.85
}
```

### 2. Matching Engine Integration
**File**: `app/services/matching_engine_service.py`

**Method**: `process_voter_submission_llm()`

**Flow**:
1. Fetch all candidates for election
2. Filter to candidates with responses
3. For each candidate:
   - Call LLM direct matching service
   - Convert LLM result to `CandidateMatchSchema`
4. Sort by match percentage
5. Assign categories (TOP/OTHER/UNMATCH)
6. Return `MatchResultsResponseSchema`

**Schema Conversion**:
- LLM returns raw match data
- Converted to `CandidateMatchSchema` (API format)
- Assembled into `MatchResultsResponseSchema`

### 3. API Route
**File**: `app/api/routes/matching_engine.py`

Changed from:
```python
results = await matching_engine.process_voter_submission(submission)
```

To:
```python
results = await matching_engine.process_voter_submission_llm(submission)
```

## Benefits

1. **No Caching Issues**: No dimension discovery cache = no stale mappings
2. **Intelligent Matching**: LLM understands context and nuance
3. **Flexible**: Works even when voter/candidate answer different questions
4. **Simple**: One LLM call per candidate vs complex pipeline

## Configuration

Uses OpenAI GPT-4o model via `settings.OPENAI_API_KEY`

## Error Handling

- Failed candidates get 0% match with error explanation
- All errors logged with candidate ID
- Response structure always valid even on errors

## Backwards Compatibility

✅ Request payload structure unchanged
✅ Response structure unchanged
✅ All existing API consumers work without changes
