import inspect

from app.services.matching_engine_service import MatchingEngine


class TestImplementationDetails:
    """Test to examine the implementation details of the matching engine."""

    def test_inspect_ranking_similarity_implementation(self):
        """Inspect the implementation of the ranking similarity calculation."""
        # Create the matching engine
        engine = MatchingEngine()

        # Get the source code of the _calculate_similarity method
        source = inspect.getsource(engine._calculate_similarity)

        # Print for debugging
        print("\n\n=== _calculate_similarity method ===")
        print(source)

        # Extract the section for ranking similarity
        ranking_section = ""
        capture = False
        for line in source.split('\n'):
            if 'elif response_type == \'ranking\':' in line:
                capture = True
            elif capture and line.strip().startswith('elif '):
                capture = False
                break

            if capture:
                ranking_section += line + '\n'

        print("\n\n=== Ranking Similarity Implementation ===")
        print(ranking_section)

        # Testing the actual implementation directly
        # Create test data
        voter_ranking = {"rank1": "Option 1", "rank2": "Option 2", "rank3": "Option 3"}
        candidate_ranking_same = {"rank1": "Option 1", "rank2": "Option 2", "rank3": "Option 3"}
        candidate_ranking_different = {"rank1": "Option 4", "rank2": "Option 5", "rank3": "Option 6"}
        candidate_ranking_partial = {"rank1": "Option 1", "rank2": "Option 5", "rank3": "Option 6"}

        # Test the internal logic directly
        # First, extract the top 3 choices as the implementation seems to do
        voter_top = list(voter_ranking.keys())[:3]
        candidate_same_top = list(candidate_ranking_same.keys())[:3]
        candidate_diff_top = list(candidate_ranking_different.keys())[:3]
        candidate_partial_top = list(candidate_ranking_partial.keys())[:3]

        # Print what these contain
        print("\n\n=== Test Data ===")
        print(f"Voter top: {voter_top}")
        print(f"Candidate same top: {candidate_same_top}")
        print(f"Candidate diff top: {candidate_diff_top}")
        print(f"Candidate partial top: {candidate_partial_top}")

        # Check common items for same
        common_same = set(voter_top).intersection(set(candidate_same_top))
        similarity_same = len(common_same) / max(len(voter_top), len(candidate_same_top))
        print(f"\nCommon items (same): {common_same}, similarity: {similarity_same}")

        # Check common items for different
        common_diff = set(voter_top).intersection(set(candidate_diff_top))
        similarity_diff = len(common_diff) / max(len(voter_top), len(candidate_diff_top))
        print(f"Common items (diff): {common_diff}, similarity: {similarity_diff}")

        # Check common items for partial
        common_partial = set(voter_top).intersection(set(candidate_partial_top))
        similarity_partial = len(common_partial) / max(len(voter_top), len(candidate_partial_top))
        print(f"Common items (partial): {common_partial}, similarity: {similarity_partial}")

        # Next, check what would happen if we used values instead of keys
        voter_top_values = [voter_ranking[k] for k in list(voter_ranking.keys())[:3]]
        candidate_same_values = [candidate_ranking_same[k] for k in list(candidate_ranking_same.keys())[:3]]
        candidate_diff_values = [candidate_ranking_different[k] for k in list(candidate_ranking_different.keys())[:3]]

        print(f"\nVoter top values: {voter_top_values}")
        print(f"Candidate same values: {candidate_same_values}")
        print(f"Candidate diff values: {candidate_diff_values}")

        # Check common items for same (values)
        common_same_val = set(voter_top_values).intersection(set(candidate_same_values))
        similarity_same_val = len(common_same_val) / max(len(voter_top_values), len(candidate_same_values))
        print(f"Common items (same values): {common_same_val}, similarity: {similarity_same_val}")

        # Check common items for different (values)
        common_diff_val = set(voter_top_values).intersection(set(candidate_diff_values))
        similarity_diff_val = len(common_diff_val) / max(len(voter_top_values), len(candidate_diff_values))
        print(f"Common items (diff values): {common_diff_val}, similarity: {similarity_diff_val}")

        # Run the actual method to see the results
        print("\n\n=== Actual Method Results ===")
        similarity_same, explanation_same = engine._calculate_similarity(voter_ranking, candidate_ranking_same, "ranking")
        print(f"Same: {similarity_same}, {explanation_same}")

        similarity_diff, explanation_diff = engine._calculate_similarity(voter_ranking, candidate_ranking_different, "ranking")
        print(f"Different: {similarity_diff}, {explanation_diff}")

        similarity_partial, explanation_partial = engine._calculate_similarity(voter_ranking, candidate_ranking_partial, "ranking")
        print(f"Partial: {similarity_partial}, {explanation_partial}")
        