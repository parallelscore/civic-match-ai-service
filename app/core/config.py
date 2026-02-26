import os
from typing import List, ClassVar, Optional

from dotenv import load_dotenv
from pydantic import Field, ConfigDict, field_validator
from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    PROJECT_NAME: str = Field('Civic Match Matching Engine', json_schema_extra={'env': 'PROJECT_NAME'})
    DESCRIPTION: str = Field('This is the backend service for Civic Match Matching Engine',
                             json_schema_extra={'env': 'DESCRIPTION'})
    VERSION: str = Field('2.0.0', json_schema_extra={'env': 'VERSION'})
    CORS_ORIGINS: List[str] = Field(default=['*'], json_schema_extra={'env': 'CORS_ORIGINS'})
    API_V1_STR: str = Field('/api/v2', json_schema_extra={'env': 'API_V1_STR'})

    POSTGRESQL_DATABASE_URL: str = Field(..., json_schema_extra={'env': 'POSTGRESQL_DATABASE_URL'})

    AI_SERVICE_API_URL: str = Field(..., json_schema_extra={'env': 'AI_SERVICE_API_URL'})
    BACKEND_API_URL: str = Field(..., json_schema_extra={'env': 'BACKEND_API_URL'})
    USE_MOCK_BACKEND_API_URL: bool = Field(False, json_schema_extra={'env': 'USE_MOCK_BACKEND_API'})
    MOCK_BACKEND_API_URL: str = Field(..., json_schema_extra={'env': 'MOCK_BACKEND_API'})

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, json_schema_extra={'env': 'OPENAI_API_KEY'})
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, json_schema_extra={'env': 'ANTHROPIC_API_KEY'})
    LLM_PROVIDER: str = Field(default='openai', json_schema_extra={'env': 'LLM_PROVIDER'})  # 'openai' or 'anthropic'
    LLM_MODEL: str = Field(default='gpt-3.5-turbo', json_schema_extra={'env': 'LLM_MODEL'})
    LLM_MAX_TOKENS: int = Field(default=2000, json_schema_extra={'env': 'LLM_MAX_TOKENS'})
    LLM_TEMPERATURE: float = Field(default=0.1, json_schema_extra={'env': 'LLM_TEMPERATURE'})

    # Embedding Configuration
    EMBEDDING_MODEL: str = Field(default='all-MiniLM-L6-v2', json_schema_extra={'env': 'EMBEDDING_MODEL'})
    EMBEDDING_SIMILARITY_THRESHOLD: float = Field(default=0.65, json_schema_extra={'env': 'EMBEDDING_SIMILARITY_THRESHOLD'})

    # Caching Configuration
    REDIS_DATABASE_URL: Optional[str] = Field(default=None, json_schema_extra={'env': 'REDIS_DATABASE_URL'})
    CACHE_TTL_SECONDS: int = Field(default=3600, json_schema_extra={'env': 'CACHE_TTL_SECONDS'})  # 1 hour

    # Matching Configuration
    ENABLE_LLM_MATCHING: bool = Field(default=True, json_schema_extra={'env': 'ENABLE_LLM_MATCHING'})
    ENABLE_SEMANTIC_MATCHING: bool = Field(default=True, json_schema_extra={'env': 'ENABLE_SEMANTIC_MATCHING'})
    LLM_RETRY_ATTEMPTS: int = Field(default=3, json_schema_extra={'env': 'LLM_RETRY_ATTEMPTS'})
    LLM_TIMEOUT_SECONDS: int = Field(default=30, json_schema_extra={'env': 'LLM_TIMEOUT_SECONDS'})

    # Debug Configuration
    # When True, the /matching_engine/debug endpoint is registered and accessible.
    # Must be False (default) in production to prevent exposing internal details.
    # Set DEBUG_MODE=true in your .env file to enable during development.
    DEBUG_MODE: bool = Field(default=False, json_schema_extra={'env': 'DEBUG_MODE'})

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        # Treat empty strings as missing/unset, allowing defaults to be used
        str_strip_whitespace=True,
    )

    @field_validator(
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'REDIS_DATABASE_URL',
        mode='before'
    )
    @classmethod
    def empty_str_to_none_optional(cls, v):
        """Convert empty strings to None for optional string fields"""
        if v == '':
            return None
        return v
    
    @field_validator(
        'LLM_MAX_TOKENS',
        mode='before'
    )
    @classmethod
    def validate_llm_max_tokens(cls, v):
        """Handle empty string for LLM_MAX_TOKENS"""
        if v == '' or v is None:
            return 2000
        return int(v)
    
    @field_validator(
        'CACHE_TTL_SECONDS',
        mode='before'
    )
    @classmethod
    def validate_cache_ttl(cls, v):
        """Handle empty string for CACHE_TTL_SECONDS"""
        if v == '' or v is None:
            return 3600
        return int(v)
    
    @field_validator(
        'LLM_RETRY_ATTEMPTS',
        mode='before'
    )
    @classmethod
    def validate_retry_attempts(cls, v):
        """Handle empty string for LLM_RETRY_ATTEMPTS"""
        if v == '' or v is None:
            return 3
        return int(v)
    
    @field_validator(
        'LLM_TIMEOUT_SECONDS',
        mode='before'
    )
    @classmethod
    def validate_timeout(cls, v):
        """Handle empty string for LLM_TIMEOUT_SECONDS"""
        if v == '' or v is None:
            return 30
        return int(v)
    
    @field_validator(
        'LLM_TEMPERATURE',
        mode='before'
    )
    @classmethod
    def validate_temperature(cls, v):
        """Handle empty string for LLM_TEMPERATURE"""
        if v == '' or v is None:
            return 0.1
        return float(v)
    
    @field_validator(
        'EMBEDDING_SIMILARITY_THRESHOLD',
        mode='before'
    )
    @classmethod
    def validate_similarity_threshold(cls, v):
        """Handle empty string for EMBEDDING_SIMILARITY_THRESHOLD"""
        if v == '' or v is None:
            return 0.65
        return float(v)
    
    @field_validator(
        'ENABLE_LLM_MATCHING',
        mode='before'
    )
    @classmethod
    def validate_enable_llm(cls, v):
        """Handle empty string for ENABLE_LLM_MATCHING"""
        if v == '' or v is None:
            return True
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)
    
    @field_validator(
        'ENABLE_SEMANTIC_MATCHING',
        mode='before'
    )
    @classmethod
    def validate_enable_semantic(cls, v):
        """Handle empty string for ENABLE_SEMANTIC_MATCHING"""
        if v == '' or v is None:
            return True
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)

    @field_validator('DEBUG_MODE', mode='before')
    @classmethod
    def validate_debug_mode(cls, v):
        """Handle empty string for DEBUG_MODE"""
        if v == '' or v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)
    
    @field_validator(
        'USE_MOCK_BACKEND_API_URL',
        mode='before'
    )
    @classmethod
    def validate_use_mock(cls, v):
        """Handle empty string for USE_MOCK_BACKEND_API_URL"""
        if v == '' or v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)


class DevConfig(BaseConfig):
    DEBUG: bool = Field(True, json_schema_extra={'env': 'DEBUG'})


class DemoConfig(BaseConfig):
    DEBUG: bool = Field(True, json_schema_extra={'env': 'DEBUG'})


class ProdConfig(BaseConfig):
    DEBUG: bool = Field(False, json_schema_extra={'env': 'DEBUG'})


def get_settings():
    env = os.getenv('ENV', '').lower()

    env_mapping = {
        'prod': ('.env.prod', ProdConfig),
        'demo': ('.env.demo', DemoConfig),
        'dev': ('.env.dev', DevConfig),
    }

    # If ENV is not specified, default to the basic .env file
    if not env:
        load_dotenv('.env')
        return BaseConfig()

    # Load the environment-specific .env file if ENV is specified
    env_file, config_class = env_mapping.get(env, ('.env', BaseConfig))

    # Load the env file only if it exists
    if os.path.exists(env_file):
        print(f'Loading {env} configuration from {env_file}')
        load_dotenv(env_file)
    else:
        print(f'Environment file {env_file} does not exist')

    return config_class()


settings = get_settings()
