from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Text
from fastapi import Depends
from sqlmodel import create_engine, SQLModel, Session, select, Field, Relationship
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

class Conversation(SQLModel, table=True):
    id: int = Field(default = None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index = True)
    title: Optional[str] = Field(default=None, max_length=200)
    created_at: datetime = Field(
        default_factory=get_utc_now,
        sa_type=DateTime(timezone=True)
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_type=DateTime(timezone=True)
    )

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)
    role: str = Field(max_length=20)
    content: str = Field(sa_column=Column(Text))
    sources: Optional[str] = Field(default=None, sa_column=Column(Text))
    retrieval_time: Optional[float] = Field(default=None)
    generation_time: Optional[float] = Field(default=None)
    created_at: datetime = Field(
        default_factory=get_utc_now,
        sa_type=DateTime(timezone=True)
    )
    

DATABASE_URL = settings.database_url
engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]


