from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "SkinScan API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_PREFIX: str = "/api"

    # Inference Settings
    USE_MOCK_INFERENCE: bool = False
    DEFAULT_MODEL: str = "b3"
    MODEL_WEIGHTS_DIR: str = "weights"
    MODEL_DEVICE: str = "cpu"
    HF_TIMEOUT_SECONDS: int = 180

    # Upload Settings
    MAX_UPLOAD_SIZE_MB: int = 10

    # Security Settings
    FRONTEND_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"

    @property
    def origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.FRONTEND_ORIGINS.split(",")]

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str) and value.lower() in {"release", "prod", "production"}:
            return False
        return value

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
