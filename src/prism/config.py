from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    prism_host: str = Field(default="0.0.0.0")
    prism_port: int = Field(default=9877)
    prism_api_key: str = Field(default="")

    upstream_base_url: str = Field(default="")
    upstream_api_key: str = Field(default="")

    default_model: str = Field(default="")

    # Access control
    clients_file: str = Field(default="clients.json")

    # Usage tracking
    usage_db: str = Field(default="prism_usage.db")

    # Admin API
    prism_admin_key: str = Field(default="")


settings = Settings()
