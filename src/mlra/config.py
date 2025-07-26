"""App configuration loaded from environment / .env file."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"


settings = Settings()
