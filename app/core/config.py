"""Application settings with explicit offline and real-mode boundaries."""

from decimal import Decimal
from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AppMode = Literal["fake", "local", "real"]


class Settings(BaseSettings):
    """Runtime settings.

    ``fake`` and ``local`` never imply permission to call external services.
    ``real`` fails during validation unless the required LLM credential exists.
    Dependencies are still connected lazily by later-stage adapters.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Enterprise Agentic RAG"
    app_env: Literal["local", "test", "production"] = "local"
    app_mode: AppMode = "fake"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    bind_host: str = "127.0.0.1"

    dev_auth_enabled: bool = False
    dev_auth_secret: SecretStr | None = None
    dev_auth_issuer: str = "enterprise-agentic-rag-dev"
    dev_auth_audience: str = "enterprise-agentic-rag"

    llm_provider: Literal["deepseek", "openai_compatible"] = "deepseek"
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = Field(default="deepseek-flash", min_length=1)
    llm_api_key: SecretStr | None = None
    llm_structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_max_attempts: int = Field(default=3, ge=1, le=5)
    llm_max_concurrency: int = Field(default=4, ge=1, le=64)
    llm_max_calls: int = Field(default=12, ge=1, le=10_000)
    llm_max_total_tokens: int | None = Field(default=None, ge=1)
    llm_max_output_tokens: int = Field(default=1_024, ge=1, le=131_072)
    llm_input_price_per_million: Decimal | None = Field(default=None, ge=0)
    llm_output_price_per_million: Decimal | None = Field(default=None, ge=0)

    database_url: SecretStr | None = None
    test_database_url: SecretStr | None = None
    redis_url: SecretStr | None = None
    milvus_uri: str | None = None
    test_milvus_uri: str | None = None
    milvus_collection: str = "enterprise_rag_hybrid_chunks"
    embedding_dimension: int = 128
    retrieval_dense_top_k: int = Field(default=30, ge=1, le=16_384)
    retrieval_lexical_top_k: int = Field(default=30, ge=1, le=16_384)
    retrieval_rrf_k: int = Field(default=60, ge=1)
    retrieval_fusion_top_k: int = Field(default=20, ge=1, le=16_384)
    reranker_candidate_limit: int = Field(default=20, ge=5, le=100)
    reranker_top_k: int = Field(default=8, ge=5, le=8)
    reranker_timeout_seconds: float = Field(default=2.0, gt=0, le=60)
    context_token_budget: int = Field(default=1_200, ge=1, le=100_000)
    context_max_parents: int = Field(default=8, ge=1, le=100)

    @model_validator(mode="after")
    def validate_real_mode(self) -> Self:
        """Prevent an accidental real-mode start without an LLM credential."""
        if self.reranker_candidate_limit < self.reranker_top_k:
            raise ValueError("RERANKER_CANDIDATE_LIMIT cannot be smaller than RERANKER_TOP_K")
        if self.app_mode == "real" and self.llm_api_key is None:
            raise ValueError("LLM_API_KEY is required when APP_MODE=real")
        if (self.llm_input_price_per_million is None) != (
            self.llm_output_price_per_million is None
        ):
            raise ValueError("both LLM token prices must be set together")
        if self.dev_auth_enabled:
            if self.app_env != "local" or self.app_mode == "real":
                raise ValueError("DEV_AUTH_ENABLED is restricted to local non-real mode")
            if self.bind_host not in {"127.0.0.1", "::1", "localhost"}:
                raise ValueError("DEV_AUTH_ENABLED requires a loopback BIND_HOST")
            if self.dev_auth_secret is None:
                raise ValueError("DEV_AUTH_SECRET is required when DEV_AUTH_ENABLED=true")
            if len(self.dev_auth_secret.get_secret_value()) < 32:
                raise ValueError("DEV_AUTH_SECRET must contain at least 32 characters")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached process settings without creating network clients."""
    return Settings()
