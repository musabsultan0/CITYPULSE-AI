from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    app_name: str = "CITYPULSE AI"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = f"sqlite:///{ROOT_DIR.as_posix()}/data/citypulse.db"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
    cors_origin_regex: str = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    open_meteo_enabled: bool = False
    latitude: float = 17.3850
    longitude: float = 78.4867
    traffic_api_enabled: bool = False
    traffic_api_provider: str = "tomtom"
    traffic_api_key: str = ""
    traffic_api_timeout_seconds: float = 10.0
    traffic_model_path: str = str(ROOT_DIR / "ml" / "traffic" / "models" / "traffic_model.joblib")
    flood_model_path: str = str(ROOT_DIR / "ml" / "flood" / "models" / "flood_model.joblib")
    flood_metrics_path: str = str(ROOT_DIR / "ml" / "flood" / "models" / "flood_metrics.json")
    rag_mode: str = "faiss"
    rag_persist_directory: str = "./data/vector_store"
    embedding_provider: str = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 80
    llm_enabled: bool = False
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 20.0
    model_config = SettingsConfigDict(env_file=str(ROOT_DIR / ".env"), extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

settings = Settings()
