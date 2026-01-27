from datetime import datetime, timezone
from sqlalchemy import DateTime
from fastapi import Depends
from sqlmodel import create_engine, SQLModel, Session, select, Field
from typing import Annotated, Optional
from app.config import settings


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)

class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    email: str = Field(nullable=False, unique=True)
    password: str = Field(nullable=False)
    created_at: datetime = Field(
        default_factory=get_utc_now, nullable=False,
        sa_type = DateTime(timezone=True)
    )
    

DATABASE_URL = settings.database_url
engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]


