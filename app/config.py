from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int

    class Config:
        env_file = ".env"
        extra = "ignore"


class VectorDbSettings(BaseSettings):
    faiss_index_path: str
    embeddings_path: str
    chunks_path: str
    index_path: str

    class Config:
        env_file = ".env"
        extra = "ignore"

class embedding_model(BaseSettings):
    model_name: str = "pritamdeka/S-BlueBERT-snli-multinli-stsb"
    embedding_dimension: int = 768
    batch_size: int = 32

    class Config:
        env_file = ".env"
        extra = "ignore"


class ReRanker(BaseSettings):
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    default_k_per_query: int = 3
    max_candidates: int = 10
    final_top_k: int = 5


class LLM_Setting(BaseSettings):
    llm_model_name: str = 'meta-llama/Llama-3.2-3B-Instruct'
    hf_token: Optional[str] = 'bbcc'
    default_temperature: float = 0.1
    default_max_tokens: int = 256
    max_content_chars: int = 1000


settings = Settings()

vector_db_settings = VectorDbSettings()

embedding_model_settings = embedding_model()

reranker_settings = ReRanker()

llm_settings = LLM_Setting()