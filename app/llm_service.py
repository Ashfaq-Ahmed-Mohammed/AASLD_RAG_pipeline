"""
LLM service for generating streaming responses.
Uses HuggingFace Inference API with chat completion for Llama models.
"""

import asyncio
from typing import AsyncGenerator, Dict, Optional, List
import logging
from huggingface_hub import AsyncInferenceClient
from textwrap import dedent

from app.config import llm_settings

logger = logging.getLogger(__name__)



llm_client: Optional[AsyncInferenceClient] = None




def initialize_llm_client() -> None:
    """Initialize HuggingFace Inference API client."""
    global llm_client
    
    if llm_client is not None:
        logger.info("LLM client already initialized")
        return
    
    logger.info("Initializing LLM client...")
    
    try:
        llm_client = AsyncInferenceClient(
            model=llm_settings.llm_model_name,
            token=llm_settings.hf_token
        )
        
        logger.info(f"✓ LLM client initialized: {llm_settings.llm_model_name}")
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        raise RuntimeError(f"LLM initialization failed: {e}")




def build_prompt(context: str, question: str) -> str:
    """Build prompt for medical RAG system."""
    prompt = dedent(f"""
    You are a medical information assistant. Answer the question ONLY using the context below.
    If the answer is not present, say "This information is not available in the provided documents."
    Do not provide diagnosis or treatment advice. Be concise and include sources.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """).strip()
    
    return prompt


def build_chat_messages(context: str, question: str) -> List[Dict[str, str]]:
    """
    Build chat messages for Llama models.
    Llama uses conversational format instead of plain text.
    """
    system_message = (
        "You are a medical information assistant. Answer questions ONLY using the provided context. "
        "If the answer is not in the context, say 'This information is not available in the provided documents.' "
        "Do not provide diagnosis or treatment advice. Be concise."
    )
    
    user_message = dedent(f"""
    Context:
    {context}
    
    Question: {question}
    """).strip()
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    return messages



def render_context(docs: list[Dict], max_chars: int = 1000) -> str:
    """Render retrieved chunks into context string."""
    parts = []
    
    for doc in docs:
        metadata = doc.get('metadata', {})
        
        heading = metadata.get('heading', 'Unknown')
        source = metadata.get('source', 'Unknown')
        text = doc.get('text', '')[:max_chars]
        
        section = f"Section: {heading} (Source: {source})\n{text}"
        parts.append(section)
    
    return "\n\n".join(parts)




async def stream_llm_response(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 256
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response using chat completion API.
    Works with Llama and other chat models.
    """
    global llm_client
    
    if llm_client is None:
        raise RuntimeError("LLM client not initialized")
    
    try:
        logger.debug(f"Starting chat completion (temp={temperature}, max_tokens={max_tokens})")
        
    
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        
        stream = await llm_client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
       
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        
        logger.debug("Generation completed")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        yield f"\n\n[Error: Generation failed - {str(e)}]"




async def generate_response(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 256
) -> str:
    """Generate complete response without streaming."""
    global llm_client
    
    if llm_client is None:
        raise RuntimeError("LLM client not initialized")
    
    try:
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
       
        response = await llm_client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
       
        if response.choices and len(response.choices) > 0:
            answer = response.choices[0].message.content.strip()
            return answer
        else:
            return "[Error: No response generated]"
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"[Error: Generation failed - {str(e)}]"




def get_llm_info() -> Dict:
    """Get LLM client information."""
    if llm_client is None:
        return {
            'initialized': False,
            'message': 'LLM client not initialized'
        }
    
    return {
        'initialized': True,
        'model': llm_settings.llm_model_name,
        'provider': 'HuggingFace Inference API',
        'streaming': True,
        'max_tokens': llm_settings.default_max_tokens
    }
