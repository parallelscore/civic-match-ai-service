import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.utils.logging_util import setup_logger
from app.schemas.voters_schema import QuestionSimilaritySchema


class SemanticMatchingService:
    """Service for semantic matching using sentence embeddings."""

    def __init__(self):
        self.logger = setup_logger(__name__)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self.logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            self.model = None

    def encode_questions(self, questions: List[str]) -> Optional[np.ndarray]:
        """Encode questions into embeddings."""
        if not self.model or not questions:
            return None

        try:
            embeddings = self.model.encode(questions, convert_to_numpy=True)
            self.logger.debug(f"Encoded {len(questions)} questions into embeddings")
            return embeddings
        except Exception as e:
            self.logger.error(f"Failed to encode questions: {str(e)}")
            return None

    def find_similar_questions(self, voter_question: str, candidate_questions: List[str],
                               threshold: float = None) -> List[QuestionSimilaritySchema]:
        """
        Find semantically similar questions using embeddings.
        """
        if not self.model or not candidate_questions:
            return []

        threshold = threshold or settings.EMBEDDING_SIMILARITY_THRESHOLD

        try:
            # Encode all questions
            all_questions = [voter_question] + candidate_questions
            embeddings = self.encode_questions(all_questions)

            if embeddings is None:
                return []

            # Calculate similarities
            voter_embedding = embeddings[0:1]  # First embedding
            candidate_embeddings = embeddings[1:]  # Rest of embeddings

            similarities = cosine_similarity(voter_embedding, candidate_embeddings)[0]

            # Create similarity objects for matches above threshold
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    result = QuestionSimilaritySchema(
                        voter_question=voter_question,
                        candidate_question=candidate_questions[i],
                        similarity_score=float(similarity),
                        similarity_method="embedding",
                        explanation=f"Semantic similarity: {similarity:.3f}"
                    )
                    results.append(result)

            # Sort by similarity score (highest first)
            results.sort(key=lambda x: x.similarity_score, reverse=True)

            self.logger.debug(f"Found {len(results)} similar questions above threshold {threshold}")
            return results

        except Exception as e:
            self.logger.error(f"Semantic similarity calculation failed: {str(e)}")
            return []

    def batch_find_similar_questions(self, voter_questions: List[str],
                                     candidate_questions: List[str],
                                     threshold: float = None) -> Dict[str, List[QuestionSimilaritySchema]]:
        """
        Find similar questions for multiple voter questions efficiently.
        """
        if not self.model or not voter_questions or not candidate_questions:
            return {}

        threshold = threshold or settings.EMBEDDING_SIMILARITY_THRESHOLD

        try:
            # Encode all questions once
            voter_embeddings = self.encode_questions(voter_questions)
            candidate_embeddings = self.encode_questions(candidate_questions)

            if voter_embeddings is None or candidate_embeddings is None:
                return {}

            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(voter_embeddings, candidate_embeddings)

            results = {}
            for i, voter_question in enumerate(voter_questions):
                similarities = similarity_matrix[i]
                question_results = []

                for j, similarity in enumerate(similarities):
                    if similarity >= threshold:
                        result = QuestionSimilaritySchema(
                            voter_question=voter_question,
                            candidate_question=candidate_questions[j],
                            similarity_score=float(similarity),
                            similarity_method="embedding",
                            explanation=f"Semantic similarity: {similarity:.3f}"
                        )
                        question_results.append(result)

                # Sort by similarity score
                question_results.sort(key=lambda x: x.similarity_score, reverse=True)
                results[voter_question] = question_results

            self.logger.info(f"Batch processed {len(voter_questions)} voter questions against {len(candidate_questions)} candidate questions")
            return results

        except Exception as e:
            self.logger.error(f"Batch semantic similarity calculation failed: {str(e)}")
            return {}

    def get_question_clusters(self, questions: List[str],
                              num_clusters: int = None) -> Dict[int, List[str]]:
        """
        Cluster questions by semantic similarity.
        """
        if not self.model or len(questions) < 2:
            return {}

        try:
            from sklearn.cluster import KMeans

            # Encode questions
            embeddings = self.encode_questions(questions)
            if embeddings is None:
                return {}

            # Determine number of clusters
            if num_clusters is None:
                num_clusters = min(max(2, len(questions) // 3), 8)

            # Perform clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Group questions by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(questions[i])

            self.logger.info(f"Clustered {len(questions)} questions into {len(clusters)} groups")
            return clusters

        except Exception as e:
            self.logger.error(f"Question clustering failed: {str(e)}")
            return {}


# Create global instance
semantic_service = SemanticMatchingService()