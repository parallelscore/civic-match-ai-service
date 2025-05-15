import os
import sys
import json
import asyncio

# Add the project root to a Python path for imports to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.schemas.voters_schema import VoterSubmissionSchema
from app.services.matching_engine_service import matching_engine


async def test_matching_engine():
    """Test the matching engine with a sample voter submission."""
    print("Loading test data...")

    # Load the test data
    with open('test_voter_submission.json', 'r') as f:
        test_data = json.load(f)

    # Convert to VoterSubmission object
    submission = VoterSubmissionSchema(**test_data)
    print(f"Loaded test submission for voter {submission.citizen_id}")

    # Print input data summary
    print(f"Election ID: {submission.election_id}")
    print(f"Number of responses: {len(submission.responses)}")
    print("Questions by category:")

    # Group questions by category for display
    categories = {}
    for resp in submission.responses:
        cat = resp.category or "Uncategorized"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(resp.question)

    for cat, questions in categories.items():
        print(f"  {cat} ({len(questions)} questions)")
        for q in questions:
            print(f"    - {q[:60]}..." if len(q) > 60 else f"    - {q}")

    # Process the submission
    print("\nProcessing submission...")
    results = await matching_engine.process_voter_submission(submission)

    # Print the results
    print("\nMatching Results:")
    print(f"Voter ID: {results.voter_id}")
    print(f"Election ID: {results.election_id}")
    print(f"Generated at: {results.generated_at}")
    print(f"Number of matches: {len(results.matches)}")

    # Print sorted matches
    for i, match in enumerate(results.matches, 1):
        print(f"\nMatch #{i}: {match.candidate_name} (ID: {match.candidate_id})")
        print(f"Title: {match.candidate_title}")
        print(f"Match Percentage: {match.match_percentage}%")
        print(f"Top Aligned Issues: {', '.join(match.top_aligned_issues)}")

        # Print issue matches
        print("\nIssue Details:")
        for detail in match.issue_matches:
            print(f"  {detail.issue}: {detail.alignment}")
            print(f"    Your Position: {detail.voter_position}")
            print(f"    Candidate Position: {detail.candidate_position}")
            print("")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_matching_engine())
