# tests/conftest.py
#
# Global test configuration.
# Sets required environment variables BEFORE any app module is imported,
# so that pydantic-settings can boot without a real .env file.
# These are test-only dummy values — they do not connect to real services.

import os
import pytest

# ── Set required env vars before any app import ───────────────────────────────
# These must be set at module level (not inside a fixture) because pydantic
# reads them at import time when `settings = get_settings()` runs.

os.environ.setdefault("POSTGRESQL_DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
os.environ.setdefault("AI_SERVICE_API_URL", "http://localhost:8000")
os.environ.setdefault("BACKEND_API_URL", "http://localhost:8001")
os.environ.setdefault("MOCK_BACKEND_API_URL", "http://localhost:8002")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("DEBUG_MODE", "false")
os.environ.setdefault("ENV", "dev")
