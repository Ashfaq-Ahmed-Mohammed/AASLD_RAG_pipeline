from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager
from app.database.database import create_db_and_tables
from app.routers import users, auth
from app.vector_service import load_vector_database, get_vector_database_status, initialize_reranker
from app.embedding_service import initialize_embedder, get_embedding_info
import logging

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code

    logger.info("Starting up...")
    try:
        create_db_and_tables()
        load_vector_database()
        initialize_embedder()
        initialize_reranker()
    except Exception as e:
        logger.error(f"Error during startup: {e}")
    yield
    # Shutdown code
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.include_router(users.router)
app.include_router(auth.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

