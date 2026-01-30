"""
Stateful chat router with conversation history
"""

from app.database.database import SessionDep
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlmodel import select
from app.database.database import User, Message, Conversation
from app.utilities import get_password_hash
from app.models.models import (
    MessageCreate,
    ChatSettings,
    MessageResponse,
    ConversationListItem,
    ConversationDetail,
    SourceDocument
)
import logging
import time
import json
from datetime import timezone, datetime
from typing import List

from app.oauth2 import get_current_user
from app.embedding_service import embed_query
from app.vector_service import retrieve_with_reranking, render_context_string
from app.llm_service import stream_llm_response, generate_response, build_prompt



logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@router.post("/send")
async def send_message(
    message: MessageCreate,
    session: SessionDep, 
    current_user: User = Depends(get_current_user),
    settings: ChatSettings = ChatSettings()
):
    """
    Send a message and get AI response with RAG.
    
    **How it works:**
    1. Creates new conversation (if conversation_id is null) or continues existing one
    2. Retrieves last 5 messages for context
    3. Searches vector database for relevant medical documents
    4. Combines conversation history + documents
    5. Sends to LLM and streams/returns response
    6. Saves everything to PostgreSQL
    """
    
    try:
        # STEP 1: Get or create conversation
        if message.conversation_id:
            statement = select(Conversation).where(
                Conversation.id == message.conversation_id,
                Conversation.user_id == current_user.id
            )
            conversation = session.exec(statement).first()
            
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            
            conversation.updated_at = datetime.now(timezone.utc)
            session.add(conversation)
            session.commit()
        else:
            title = message.content[:50] + "..." if len(message.content) > 50 else message.content
            conversation = Conversation(
                user_id=current_user.id,
                title=title,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
        
        # STEP 2: Save user message
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=message.content,
            created_at=datetime.now(timezone.utc)
        )
        session.add(user_message)
        session.commit()
        session.refresh(user_message)
        
        # STEP 3: Get conversation history (last 5 messages)
        history_statement = select(Message).where(
            Message.conversation_id == conversation.id
        ).order_by(Message.created_at.desc()).limit(5)
        history = session.exec(history_statement).all()
        history.reverse()
        
        # STEP 4: Build history context
        history_context = ""
        for msg in history[:-1]:
            history_context += f"{msg.role.capitalize()}: {msg.content}\n"
        
        logger.info(f"User {current_user.email} sent message in conversation {conversation.id}")
        
        # STEP 5: Stream or return complete response
        if settings.stream:
            return StreamingResponse(
                stream_chat_response(
                    conversation=conversation,
                    user_message=user_message,
                    question=message.content,
                    history_context=history_context,
                    settings=settings,
                    session=session
                ),
                media_type="text/event-stream"
            )
        else:
            return await generate_complete_response(
                conversation=conversation,
                user_message=user_message,
                question=message.content,
                history_context=history_context,
                settings=settings,
                session=session
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/conversations", response_model=List[ConversationListItem])
async def list_conversations(
    session: SessionDep,  # ✅ No Depends() needed
    current_user: User = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
):
    """Get all conversations for the current user"""
    
    statement = select(Conversation).where(
        Conversation.user_id == current_user.id
    ).order_by(
        Conversation.updated_at.desc()
    ).offset(offset).limit(limit)
    
    conversations = session.exec(statement).all()
    
    result = []
    for conv in conversations:
        msg_statement = select(Message).where(Message.conversation_id == conv.id)
        messages = session.exec(msg_statement).all()
        
        result.append(ConversationListItem(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=len(messages),
            last_message=messages[-1].content[:100] if messages else None
        ))
    
    return result


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    session: SessionDep,  # ✅ No Depends() needed
    current_user: User = Depends(get_current_user)
):
    """Get a specific conversation with all messages"""
    
    statement = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    )
    conversation = session.exec(statement).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    msg_statement = select(Message).where(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at)
    messages = session.exec(msg_statement).all()
    
    message_responses = []
    for msg in messages:
        sources = None
        if msg.sources:
            try:
                sources_data = json.loads(msg.sources)
                sources = [SourceDocument(**s) for s in sources_data]
            except Exception as e:
                logger.warning(f"Failed to parse sources for message {msg.id}: {e}")
        
        message_responses.append(MessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            sources=sources,
            retrieval_time=msg.retrieval_time,
            generation_time=msg.generation_time,
            created_at=msg.created_at
        ))
    
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=message_responses
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    session: SessionDep,  # ✅ No Depends() needed
    current_user: User = Depends(get_current_user)
):
    """Delete a conversation and all its messages"""
    
    statement = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    )
    conversation = session.exec(statement).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    session.delete(conversation)
    session.commit()
    
    logger.info(f"User {current_user.email} deleted conversation {conversation_id}")
    
    return {
        "message": "Conversation deleted successfully",
        "conversation_id": conversation_id
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def stream_chat_response(
    conversation: Conversation,
    user_message: Message,
    question: str,
    history_context: str,
    settings: ChatSettings,
    session  # Regular parameter, not annotated
):
    """Stream response with Server-Sent Events"""
    
    try:
        query_embedding = embed_query(question)
        
        docs, timings = retrieve_with_reranking(
            question=question,
            query_embeddings=[query_embedding],
            k_per_query=3,
            max_retrieved=10,
            rerank_top_k=settings.top_k
        )
        
        yield f"data: {json.dumps({'type': 'conversation_id', 'data': conversation.id})}\n\n"
        yield f"data: {json.dumps({'type': 'user_message_id', 'data': user_message.id})}\n\n"
        
        sources = [
            {
                'heading': doc['metadata']['heading'],
                'source': doc['metadata']['source'],
                'score': doc.get('score', 0.0),
                'rerank_score': doc.get('rerank_score')
            }
            for doc in docs
        ]
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        yield f"data: {json.dumps({'type': 'timings', 'data': timings})}\n\n"
        
        doc_context = render_context_string(docs, max_chars=150)
        
        if history_context:
            full_context = f"""Previous conversation:
{history_context}

Relevant medical documents:
{doc_context}"""
        else:
            full_context = doc_context
        
        t_gen = time.time()
        assistant_content = ""

        prompt = build_prompt(context=full_context, question=question)
        
        async for token in stream_llm_response(
            prompt=prompt,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        ):
            assistant_content += token
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
        
        generation_time = time.time() - t_gen
        
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=assistant_content,
            sources=json.dumps(sources),
            retrieval_time=timings['retrieval'],
            generation_time=generation_time,
            created_at=datetime.now(timezone.utc)
        )
        session.add(assistant_message)
        session.commit()
        session.refresh(assistant_message)
        
        yield f"data: {json.dumps({'type': 'generation_time', 'data': generation_time})}\n\n"
        yield f"data: {json.dumps({'type': 'assistant_message_id', 'data': assistant_message.id})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"


async def generate_complete_response(
    conversation: Conversation,
    user_message: Message,
    question: str,
    history_context: str,
    settings: ChatSettings,
    session  # Regular parameter, not annotated
):
    """Generate non-streaming response"""
    
    query_embedding = embed_query(question)
    
    docs, timings = retrieve_with_reranking(
        question=question,
        query_embeddings=[query_embedding],
        k_per_query=3,
        max_retrieved=10,
        rerank_top_k=settings.top_k
    )
    
    doc_context = render_context_string(docs, max_chars=150)
    
    if history_context:
        full_context = f"""Previous conversation:
{history_context}

Relevant medical documents:
{doc_context}"""
    else:
        full_context = doc_context

    prompt = build_prompt(context=full_context, question=question)
    
    t0 = time.time()
    answer = await generate_response(
        prompt = prompt,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens
    )
    generation_time = time.time() - t0
    
    sources = [
        SourceDocument(
            heading=doc['metadata']['heading'],
            source=doc['metadata']['source'],
            score=doc.get('score', 0.0),
            rerank_score=doc.get('rerank_score')
        )
        for doc in docs
    ]
    
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        sources=json.dumps([s.dict() for s in sources]),
        retrieval_time=timings['retrieval'],
        generation_time=generation_time,
        created_at=datetime.now(timezone.utc)
    )
    session.add(assistant_message)
    session.commit()
    session.refresh(assistant_message)
    
    return {
        "conversation_id": conversation.id,
        "user_message_id": user_message.id,
        "assistant_message_id": assistant_message.id,
        "answer": answer,
        "sources": sources,
        "retrieval_time": timings['retrieval'],
        "reranking_time": timings['reranking'],
        "generation_time": generation_time
    }
