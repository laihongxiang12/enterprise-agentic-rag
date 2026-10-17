"""Application settings with explicit offline and real-mode boundaries."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import SecretStr, model_validator
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

    llm_provider: Literal["deepseek", "openai_compatible"] = "deepseek"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: SecretStr | None = None

    database_url: SecretStr | None = None
    redis_url: SecretStr | None = None
    milvus_uri: str | None = None

    @model_validator(mode="after")
    def validate_real_mode(self) -> Self:
        """Prevent an accidental real-mode start without an LLM credential."""
        if self.app_mode == "real" and self.llm_api_key is None:
            raise ValueError("LLM_API_KEY is required when APP_MODE=real")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached process settings without creating network clients."""
    return Settings()
