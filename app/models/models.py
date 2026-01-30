from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional


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

class DocumentCreate(BaseModel):
    heading: str
    source: str
    score: Optional[float] = None

class ChatResponse(BaseModel):
    response_text: str
    source_documents: Optional[list[DocumentCreate]] = None


class MessageCreate(BaseModel):
    
    conversation_id: Optional[int] = None  
    content: str = Field(..., min_length=1, max_length=2000)

class ChatSettings(BaseModel):
    top_k: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=50, le=512)
    stream: bool = Field(default=True)

class SourceDocument(BaseModel):
    """Retrieved source document"""
    heading: str
    source: str
    score: float
    rerank_score: Optional[float] = None


class MessageResponse(BaseModel):
    """Response: Single message"""
    id: int
    role: str
    content: str
    sources: Optional[List[SourceDocument]] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class ConversationListItem(BaseModel):
    """Response: Conversation in list"""
    id: int
    title: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    message_count: int
    last_message: Optional[str]


class ConversationDetail(BaseModel):
    """Response: Full conversation with messages"""
    id: int
    title: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    messages: List[MessageResponse]
    
    class Config:
        from_attributes = True