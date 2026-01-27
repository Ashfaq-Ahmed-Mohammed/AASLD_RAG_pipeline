from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserRead(BaseModel):
    id: int
    email: EmailStr
    created_at: Optional[datetime] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    id: Optional[str] = None


class Query(BaseModel):
    query_text: str
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.0

class DocumentCreate(BaseModel):
    heading: str
    source: str
    score: Optional[float] = None

class ChatResponse(BaseModel):
    response_text: str
    source_documents: Optional[list[DocumentCreate]] = None

