"""Configuration helpers."""
# app/core/config.py
from typing import List, Optional, Literal
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn, AnyUrl, model_validator
from typing import Literal, Optional

# project root detection
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Provide settings behavior."""
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), extra="ignore")

    ENV: Literal["local", "dev", "staging", "prod"] = "local"
    APP_NAME: str = "mindmirror-personality"
    API_V1_PREFIX: str = "/api/v1"

    # Optional — API starts without DB when not set (local/testing mode)
    #DATABASE_URL: Optional[AnyUrl] = None
    LOG_LEVEL: str = "INFO"
    JSON_LOGS: bool = True
    POSTGRES_SERVER: str = Field(default="localhost")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_DB: str = Field(default="mindmirror")

    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None
    DATABASE_URL: Optional[str] = None
    DB_CONNECT_TIMEOUT_SECONDS: float = 30.0

    CREATE_TABLES_ON_STARTUP: bool = True

    @staticmethod
    def _normalize_database_url(raw_url: str) -> str:
        """Normalize database URL."""
        parts = urlsplit(raw_url)
        scheme = parts.scheme

        # Accept standard Postgres URLs and normalize for async runtime.
        if scheme in {"postgres", "postgresql"}:
            scheme = "postgresql+asyncpg"

        # Supabase often publishes sslmode=require; asyncpg expects ssl=require.
        query_items = parse_qsl(parts.query, keep_blank_values=True)
        has_ssl = any(key == "ssl" for key, _ in query_items)
        normalized_query_items: list[tuple[str, str]] = []
        for key, value in query_items:
            if key == "sslmode" and value == "require" and not has_ssl:
                normalized_query_items.append(("ssl", "require"))
                continue
            normalized_query_items.append((key, value))

        normalized_query = urlencode(normalized_query_items)
        return urlunsplit((scheme, parts.netloc, parts.path, normalized_query, parts.fragment))

    @model_validator(mode="after")
    def assemble_db_url(self) -> "Settings":

        """Assemble database URL."""
        if self.DATABASE_URL:
            self.DATABASE_URL = self._normalize_database_url(self.DATABASE_URL)
            return self

        if not self.DATABASE_URL:

            # sqlite test mode
            if self.POSTGRES_SERVER == "memory" or self.POSTGRES_DB == "sqlite":
                self.DATABASE_URL = "sqlite+aiosqlite:///:memory:"

            else:
                self.DATABASE_URL = (
                    f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                    f"@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"
                )

        return self

    # --------------------------------------------------
    # SECURITY
    # --------------------------------------------------

    SECRET_KEY: str = Field(default="mindmirror_secret_key", alias="JWT_SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # --------------------------------------------------
    # CORS
    # --------------------------------------------------

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "*",
    ]

    # --------------------------------------------------
    # REDDIT API
    # --------------------------------------------------

    REDDIT_CLIENT_ID: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT: str = Field(
        default="MindMirrorService/1.0.0",
        alias="REDDIT_USER_AGENT"
    )
    REDDIT_OAUTH_AUTHORIZE_URL: str = "https://www.reddit.com/api/v1/authorize"
    REDDIT_OAUTH_TOKEN_URL: str = "https://www.reddit.com/api/v1/access_token"
    REDDIT_OAUTH_ME_URL: str = "https://oauth.reddit.com/api/v1/me"
    REDDIT_OAUTH_SCOPE: str = "identity"
    REDDIT_OAUTH_DURATION: str = "permanent"
    REDDIT_OAUTH_STATE_EXP_SECONDS: int = 600
    REDDIT_OAUTH_TIMEOUT_SECONDS: float = 15.0

    # --------------------------------------------------
    # ML / ASSETS CONFIG
    # --------------------------------------------------

    MINDMIRROR_HOME: Optional[str] = Field(default=None, alias="MINDMIRROR_HOME")
    ML_ASSETS_DIR: Optional[str] = Field(default=None, alias="ML_ASSETS_DIR")
    MLFLOW_TRACKING_URI: Optional[str] = Field(
        default="http://localhost:5000",
        alias="MLFLOW_TRACKING_URI"
    )


    SECRET_KEY: str = "local-dev-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CREATE_TABLES_ON_STARTUP: bool = False

    CORS_ORIGINS: list[str] = ["http://localhost:3000", "*"]
    # Kafka settings are kept commented for rollback/reference.
    # KAFKA_BOOTSTRAP_SERVERS: str = "broker:9092"
    # KAFKA_SECURITY_PROTOCOL: str = "SSL"
    SIGNUP_SYNC_WAIT_SECONDS: float = 8.0
    # INFERENCE_CONSUMER_GROUP: str = "signup-inference-consumer"
    # INFERENCE_TOPIC_SIGNUP_REQUESTED: str = "inference.signup.requested"
    # INFERENCE_TOPIC_SIGNUP_COMPLETED: str = "inference.signup.completed"
    
    @model_validator(mode="after")
    def set_default_paths(self) -> "Settings":

        """Set default paths."""
        if self.MINDMIRROR_HOME:
            root = Path(self.MINDMIRROR_HOME).resolve()
        else:
            root = _PROJECT_ROOT

        if not self.ML_ASSETS_DIR:
            self.ML_ASSETS_DIR = str(root / "assets")

        return self


settings = Settings()

 
