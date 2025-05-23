import json
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta

from app.core.config import settings
from app.utils.logging_util import setup_logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheService:
    """Service for caching LLM and embedding results to improve performance."""

    def __init__(self):
        self.logger = setup_logger(__name__)
        self.redis_client = None
        self.memory_cache = {}  # Fallback in-memory cache
        self.cache_stats = {"hits": 0, "misses": 0}

        if REDIS_AVAILABLE and settings.REDIS_DATABASE_URL:
            self._connect_redis()

    def _connect_redis(self):
        """Connect to Redis if available."""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_DATABASE_URL,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.logger.info("Connected to Redis cache")
        except Exception as e:
            self.logger.warning(f"Failed to connect to Redis: {str(e)}. Using memory cache.")
            self.redis_client = None

    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate a consistent cache key from data."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Try Redis first
            if self.redis_client:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    self.cache_stats["hits"] += 1
                    return json.loads(cached_data)

            # Fallback to memory cache
            if key in self.memory_cache:
                cache_entry = self.memory_cache[key]
                if cache_entry["expires_at"] > datetime.now():
                    self.cache_stats["hits"] += 1
                    return cache_entry["data"]
                else:
                    # Expired entry
                    del self.memory_cache[key]

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            self.logger.error(f"Cache get error: {str(e)}")
            self.cache_stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int = None) -> bool:
        """Set value in cache."""
        ttl_seconds = ttl_seconds or settings.CACHE_TTL_SECONDS

        try:
            serialized_value = json.dumps(value)

            # Try Redis first
            if self.redis_client:
                self.redis_client.setex(key, ttl_seconds, serialized_value)
                return True

            # Fallback to memory cache
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            self.memory_cache[key] = {
                "data": value,
                "expires_at": expires_at
            }

            # Clean up expired entries periodically
            if len(self.memory_cache) > 1000:
                self._cleanup_memory_cache()

            return True

        except Exception as e:
            self.logger.error(f"Cache set error: {str(e)}")
            return False

    def _cleanup_memory_cache(self):
        """Remove expired entries from memory cache."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry["expires_at"] <= now
        ]

        for key in expired_keys:
            del self.memory_cache[key]

        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def get_or_set(self, key: str, fetch_func, ttl_seconds: int = None) -> Any:
        """Get from cache or set using fetch function."""
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value

        # Fetch new value
        new_value = await fetch_func()
        if new_value is not None:
            await self.set(key, new_value, ttl_seconds)

        return new_value

    # Specialized cache methods for common operations

    async def cache_election_topics(self, election_id: str, topics: List[Dict]) -> bool:
        """Cache discovered topics for an election."""
        key = self._generate_cache_key("election_topics", election_id)
        return await self.set(key, topics, ttl_seconds=86400)  # 24 hours

    async def get_election_topics(self, election_id: str) -> Optional[List[Dict]]:
        """Get cached topics for an election."""
        key = self._generate_cache_key("election_topics", election_id)
        return await self.get(key)

    async def cache_question_similarities(self, voter_question: str, candidate_questions: List[str],
                                          similarities: List[Dict]) -> bool:
        """Cache question similarity results."""
        cache_data = {
            "voter_question": voter_question,
            "candidate_questions": candidate_questions
        }
        key = self._generate_cache_key("question_similarities", cache_data)
        return await self.set(key, similarities, ttl_seconds=3600)  # 1 hour

    async def get_question_similarities(self, voter_question: str,
                                        candidate_questions: List[str]) -> Optional[List[Dict]]:
        """Get cached question similarity results."""
        cache_data = {
            "voter_question": voter_question,
            "candidate_questions": candidate_questions
        }
        key = self._generate_cache_key("question_similarities", cache_data)
        return await self.get(key)

    async def cache_position_alignment(self, voter_answer: str, candidate_answer: str,
                                       question: str, alignment_data: Dict) -> bool:
        """Cache position alignment analysis."""
        cache_data = {
            "voter_answer": voter_answer,
            "candidate_answer": candidate_answer,
            "question": question
        }
        key = self._generate_cache_key("position_alignment", cache_data)
        return await self.set(key, alignment_data, ttl_seconds=3600)  # 1 hour

    async def get_position_alignment(self, voter_answer: str, candidate_answer: str,
                                     question: str) -> Optional[Dict]:
        """Get cached position alignment analysis."""
        cache_data = {
            "voter_answer": voter_answer,
            "candidate_answer": candidate_answer,
            "question": question
        }
        key = self._generate_cache_key("position_alignment", cache_data)
        return await self.get(key)

    async def cache_voter_profile(self, voter_responses: List[Dict], profile: List[Dict]) -> bool:
        """Cache voter profile analysis."""
        key = self._generate_cache_key("voter_profile", voter_responses)
        return await self.set(key, profile, ttl_seconds=3600)  # 1 hour

    async def get_voter_profile(self, voter_responses: List[Dict]) -> Optional[List[Dict]]:
        """Get cached voter profile analysis."""
        key = self._generate_cache_key("voter_profile", voter_responses)
        return await self.get(key)

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate_percentage": round(hit_rate, 2),
            "total_requests": total_requests,
            "cache_type": "redis" if self.redis_client else "memory",
            "memory_cache_size": len(self.memory_cache)
        }

    async def clear_election_cache(self, election_id: str) -> bool:
        """Clear all cached data for an election."""
        try:
            if self.redis_client:
                # Use Redis pattern matching to find and delete keys
                pattern = f"*{election_id}*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    self.logger.info(f"Cleared {len(keys)} Redis cache entries for election {election_id}")

            # Clear memory cache entries
            keys_to_delete = [key for key in self.memory_cache.keys() if election_id in key]
            for key in keys_to_delete:
                del self.memory_cache[key]

            if keys_to_delete:
                self.logger.info(f"Cleared {len(keys_to_delete)} memory cache entries for election {election_id}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to clear election cache: {str(e)}")
            return False


# Create global instance
cache_service = CacheService()