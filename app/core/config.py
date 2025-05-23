import os
from typing import List, ClassVar, Optional

from dotenv import load_dotenv
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    PROJECT_NAME: str = Field('Civic Match Matching Engine', json_schema_extra={'env': 'PROJECT_NAME'})
    DESCRIPTION: str = Field('This is the backend service for Civic Match Matching Engine',
                             json_schema_extra={'env': 'DESCRIPTION'})
    VERSION: str = Field('1.0.0', json_schema_extra={'env': 'VERSION'})
    CORS_ORIGINS: List[str] = Field(default=['*'], json_schema_extra={'env': 'CORS_ORIGINS'})
    API_V1_STR: str = Field('/api/v1', json_schema_extra={'env': 'API_V1_STR'})

    POSTGRESQL_DATABASE_URL: str = Field(..., json_schema_extra={'env': 'POSTGRESQL_DATABASE_URL'})

    BACKEND_API_URL: str = Field(..., json_schema_extra={'env': 'BACKEND_API_URL'})
    USE_MOCK_BACKEND_API_URL: bool = Field(False, json_schema_extra={'env': 'USE_MOCK_BACKEND_API'})
    MOCK_BACKEND_API_URL: str = Field(..., json_schema_extra={'env': 'MOCK_BACKEND_API'})

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = Field(None, json_schema_extra={'env': 'OPENAI_API_KEY'})
    ANTHROPIC_API_KEY: Optional[str] = Field(None, json_schema_extra={'env': 'ANTHROPIC_API_KEY'})
    LLM_PROVIDER: str = Field('openai', json_schema_extra={'env': 'LLM_PROVIDER'})  # 'openai' or 'anthropic'
    LLM_MODEL: str = Field('gpt-3.5-turbo', json_schema_extra={'env': 'LLM_MODEL'})
    LLM_MAX_TOKENS: int = Field(2000, json_schema_extra={'env': 'LLM_MAX_TOKENS'})
    LLM_TEMPERATURE: float = Field(0.1, json_schema_extra={'env': 'LLM_TEMPERATURE'})

    # Embedding Configuration
    EMBEDDING_MODEL: str = Field('all-MiniLM-L6-v2', json_schema_extra={'env': 'EMBEDDING_MODEL'})
    EMBEDDING_SIMILARITY_THRESHOLD: float = Field(0.65, json_schema_extra={'env': 'EMBEDDING_SIMILARITY_THRESHOLD'})

    # Caching Configuration
    REDIS_DATABASE_URL: Optional[str] = Field(None, json_schema_extra={'env': 'REDIS_DATABASE_URL'})
    CACHE_TTL_SECONDS: int = Field(3600, json_schema_extra={'env': 'CACHE_TTL_SECONDS'})  # 1 hour

    # Matching Configuration
    ENABLE_LLM_MATCHING: bool = Field(True, json_schema_extra={'env': 'ENABLE_LLM_MATCHING'})
    ENABLE_SEMANTIC_MATCHING: bool = Field(True, json_schema_extra={'env': 'ENABLE_SEMANTIC_MATCHING'})
    LLM_RETRY_ATTEMPTS: int = Field(3, json_schema_extra={'env': 'LLM_RETRY_ATTEMPTS'})
    LLM_TIMEOUT_SECONDS: int = Field(30, json_schema_extra={'env': 'LLM_TIMEOUT_SECONDS'})

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
    )


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
